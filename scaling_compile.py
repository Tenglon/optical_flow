"""Backend x batch x resolution scaling map for the torch optical-flow solvers.

Sweeps {tvl1, deepflow} x {eager, compile, cudagraphs} x B x {240x320, 480x640}
on one GPU, reporting compile/warmup time, min-of-N ms/pair, fps and peak memory,
plus an optional torch.profiler drill-down on anomalous cells.

  srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
       --time=01:30:00 ./run_nvme.sh python scaling_compile.py --mode probe
  ... --mode sweep --algos tvl1 --backends eager compile
  ... --mode profile
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS
from benchmark_data import load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

SIZES = {"240x320": (240, 320), "480x640": (480, 640)}
BATCHES = [1, 4, 8, 16, 32, 64, 128]
REPS = 3
CELL_BUDGET_S = 300.0  # skip a cell whose projected warmup-free cost exceeds this


def make_fn(algo: str, backend: str):
    if algo == "tvl1":
        return lambda p, n: calc_flow_tvl1(p, n, backend=backend, **TVL1_PARAMS)
    return lambda p, n: calc_flow_deepflow(p, n, backend=backend, **DEEPFLOW_PARAMS)


_BASE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def load_batch(B: int, hw: tuple[int, int], dev: str):
    """B pairs on `dev`, Middlebury frames repeated to fill the batch."""
    if hw not in _BASE:
        pairs = load_pairs(max_pairs=4, gray=True, size=hw)
        if not pairs:
            raise RuntimeError("no Middlebury pairs found")
        _BASE[hw] = (
            torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs]),
            torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs]),
        )
    bp, bn = _BASE[hw]
    r = -(-B // bp.shape[0])  # ceil
    return bp.repeat(r, 1, 1)[:B].to(dev), bn.repeat(r, 1, 1)[:B].to(dev)


def raise_dynamo_limits() -> None:
    """The sweep compiles one graph per (level shape x batch); make room."""
    import torch._dynamo as dyn

    for name, val in (
        ("cache_size_limit", 8192),
        ("recompile_limit", 8192),
        ("accumulated_cache_size_limit", 65536),
        ("accumulated_recompile_limit", 65536),
    ):
        if hasattr(dyn.config, name):
            try:
                setattr(dyn.config, name, val)
            except Exception:  # noqa: BLE001 - deprecated aliases may be read-only
                pass


def dynamo_stats() -> dict:
    from torch._dynamo.utils import counters

    st = dict(counters.get("stats", {}))
    return {"unique_graphs": st.get("unique_graphs", 0),
            "frames_ok": st.get("calls_captured", 0)}


def clear_gpu() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)


@torch.no_grad()
def run_cell(algo: str, backend: str, hw: tuple[int, int], B: int, reps: int = REPS) -> dict:
    """Warm up once (timed separately), then time `reps` calls and keep the min."""
    fn = make_fn(algo, backend)
    res = f"{hw[0]}x{hw[1]}"
    rec = {"algo": algo, "res": res, "backend": backend, "B": B}
    p = n = None
    try:
        p, n = load_batch(B, hw, "cuda")
        clear_gpu()
        g0 = dynamo_stats()
        t0 = time.perf_counter()
        fn(p, n)
        torch.cuda.synchronize()
        rec["warmup_s"] = time.perf_counter() - t0
        rec["new_graphs"] = dynamo_stats()["unique_graphs"] - g0["unique_graphs"]

        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(p, n)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        rec.update(call_s=t, ms_per_pair=t / B * 1e3, fps=B / t,
                   spread=(max(ts) - t) / t,
                   peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                   status="ok")
    except Exception as exc:  # noqa: BLE001 - OOM/compile failures must not stop the sweep
        rec["status"] = "OOM" if is_oom(exc) else "ERR"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        del p, n
        clear_gpu()
    return rec


def fmt_row(r: dict) -> str:
    head = f"{r['algo']:<9} {r['res']:>8} {r['backend']:<10} {r['B']:>4}"
    if r["status"] != "ok":
        note = r.get("error", "") if r["status"] != "SKIP" else r.get("reason", "")
        return f"{head} {r['status']:>10}  {note}"
    return (f"{head} {r['warmup_s']:>9.1f} {r['ms_per_pair']:>9.2f} "
            f"{r['fps']:>8.1f} {r['peak_mib']:>9.0f} {r.get('new_graphs', 0):>7}")


HEADER = (f"{'algo':<9} {'res':>8} {'backend':<10} {'B':>4} {'warmup_s':>9} "
          f"{'ms/pair':>9} {'fps':>8} {'peakMiB':>9} {'graphs':>7}")


def emit(rec: dict) -> None:
    print(fmt_row(rec), flush=True)
    print("RESULT " + json.dumps(rec), flush=True)


def sweep(algos, backends, sizes, batches, reps) -> list[dict]:
    out: list[dict] = []
    print(HEADER, flush=True)
    for algo in algos:
        for res in sizes:
            hw = SIZES[res]
            for backend in backends:
                last = None  # (B, call_s) of the largest successful cell
                dead = None  # reason larger batches are hopeless
                for B in batches:
                    if dead:
                        rec = {"algo": algo, "res": res, "backend": backend, "B": B,
                               "status": "SKIP", "reason": dead}
                        out.append(rec)
                        emit(rec)
                        continue
                    if last is not None:
                        proj = last[1] * B / last[0] * (reps + 1)
                        if proj > CELL_BUDGET_S:
                            rec = {"algo": algo, "res": res, "backend": backend, "B": B,
                                   "status": "SKIP",
                                   "reason": f"projected {proj:.0f}s > {CELL_BUDGET_S:.0f}s"}
                            out.append(rec)
                            emit(rec)
                            dead = "budget"
                            continue
                    rec = run_cell(algo, backend, hw, B, reps)
                    out.append(rec)
                    emit(rec)
                    if rec["status"] == "ok":
                        last = (B, rec["call_s"])
                    elif rec["status"] == "OOM":
                        dead = "OOM at smaller B"
                    else:
                        dead = "error at smaller B"
    return out


@torch.no_grad()
def probe(args) -> None:
    """Cheap viability check: does cudagraphs work, and what does compile cost?"""
    print(HEADER, flush=True)
    for algo in args.algos:
        for res in args.sizes:
            for backend in args.backends:
                emit(run_cell(algo, backend, SIZES[res], args.probe_batch, reps=1))


@torch.no_grad()
def profile_cells(args) -> None:
    """Top-kernel breakdown for the cells given by --profile-batches."""
    from torch.profiler import ProfilerActivity, profile

    for algo in args.algos:
        for res in args.sizes:
            hw = SIZES[res]
            for backend in args.backends:
                for B in args.profile_batches:
                    fn = make_fn(algo, backend)
                    try:
                        p, n = load_batch(B, hw, "cuda")
                        clear_gpu()
                        t0 = time.perf_counter()
                        fn(p, n)
                        torch.cuda.synchronize()
                        warm = time.perf_counter() - t0
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        fn(p, n)
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - t0
                        with profile(activities=[ProfilerActivity.CPU,
                                                 ProfilerActivity.CUDA]) as prof:
                            fn(p, n)
                            torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001
                        print(f"\n=== {algo} {backend} B={B} {res}: "
                              f"{'OOM' if is_oom(exc) else 'ERR'} {exc}", flush=True)
                        clear_gpu()
                        continue
                    evts = prof.key_averages()
                    kern = [e for e in evts
                            if e.device_type == torch.autograd.DeviceType.CUDA]
                    launches = sum(e.count for e in kern)
                    busy = sum(e.self_device_time_total for e in kern) / 1e6
                    print(f"\n=== {algo} {backend} B={B} {res} ===", flush=True)
                    print(f"warmup {warm:.1f}s  wall {wall:.3f}s  "
                          f"ms/pair {wall / B * 1e3:.2f}  peak "
                          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
                    print(f"kernel launches {launches}  gpu busy {busy:.3f}s  "
                          f"busy/wall {busy / wall * 100:.0f}%  "
                          f"mean kernel {busy / max(launches, 1) * 1e6:.1f} us")
                    print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                                    row_limit=20,
                                                    max_name_column_width=55))
                    del p, n
                    clear_gpu()


_ORDER = {"eager": 0, "compile": 1, "cudagraphs": 2}


def load_logs(paths) -> list[dict]:
    """Collect RESULT json lines emitted by earlier (possibly parallel) runs."""
    best: dict[tuple, dict] = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("RESULT "):
                    continue
                r = json.loads(line[7:])
                k = (r["algo"], r["res"], r["backend"], r["B"])
                old = best.get(k)
                # keep the fastest successful measurement of a repeated cell
                if (old is None
                        or (r["status"] == "ok" and old["status"] != "ok")
                        or (r["status"] == "ok" == old["status"]
                            and r["ms_per_pair"] < old["ms_per_pair"])):
                    best[k] = r
    return sorted(best.values(),
                  key=lambda r: (r["algo"], r["res"], _ORDER.get(r["backend"], 9), r["B"]))


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    print("\n" + "=" * 100)
    print("FINAL TABLE")
    print(HEADER)
    for r in rows:
        print(fmt_row(r))
    if not ok:
        return
    print("\nbest (min ms/pair) per algo x res x backend:")
    seen: dict[tuple, dict] = {}
    for r in ok:
        k = (r["algo"], r["res"], r["backend"])
        if k not in seen or r["ms_per_pair"] < seen[k]["ms_per_pair"]:
            seen[k] = r
    for k in sorted(seen):
        r = seen[k]
        print(f"  {k[0]:<9} {k[1]:>8} {k[2]:<10} B={r['B']:<4} "
              f"{r['ms_per_pair']:.2f} ms/pair  {r['fps']:.1f} fps  "
              f"{r['peak_mib']:.0f} MiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sweep",
                    choices=["sweep", "probe", "profile", "report"])
    ap.add_argument("--logs", nargs="+", default=[],
                    help="report mode: logs to merge RESULT lines from")
    ap.add_argument("--algos", nargs="+", default=["tvl1", "deepflow"])
    ap.add_argument("--backends", nargs="+", default=["eager", "compile"])
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--probe-batch", type=int, default=8)
    ap.add_argument("--profile-batches", nargs="+", type=int, default=[8, 64])
    args = ap.parse_args()

    if args.mode == "report":
        summarize(load_logs(args.logs))
        return

    assert torch.cuda.is_available(), "no CUDA device"
    raise_dynamo_limits()
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"mem {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
          flush=True)
    print(f"mode={args.mode} algos={args.algos} backends={args.backends} "
          f"sizes={args.sizes} batches={args.batches} reps={args.reps}", flush=True)

    t0 = time.perf_counter()
    if args.mode == "probe":
        probe(args)
    elif args.mode == "profile":
        profile_cells(args)
    else:
        rows = sweep(args.algos, args.backends, args.sizes, args.batches, args.reps)
        summarize(rows)
    print(f"\ntotal wall: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
