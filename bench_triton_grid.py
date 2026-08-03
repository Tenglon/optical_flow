"""Speed grid: hand-written Triton kernels vs torch.compile for batched TV-L1.

Grid: {240x320, 480x640, 480x854, 720x1280} x B {1,2,4,8,16,32}
      x backend {compile, triton}, fp32, early_exit=True, median_filtering=5.
Batches are the 8 Middlebury 'other' pairs tiled up to B.

Also measures **kernel launches per primal-dual iteration** for one cell, by
profiling two runs that differ only in `inner_iterations` and dividing the
launch delta by the iteration delta -- that cancels pyramid setup, warping,
median filtering and profiler noise, leaving exactly the per-iteration cost of
the primal/dual pair (OpenCV: 2, Inductor: ~6.9).

Usage (GPU node, ~40-60 min):
    srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
         --time=00:59:00 ./run_nvme.sh python bench_triton_grid.py

    ./run_nvme.sh python bench_triton_grid.py --launch-only   # just the counts
    ./run_nvme.sh python bench_triton_grid.py --fp16          # + fp16 cells
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch

import torch_flow
from benchmark_data import load_pairs

RESOLUTIONS = [(240, 320), (480, 640), (480, 854), (720, 1280)]
BATCHES = [1, 2, 4, 8, 16, 32]
BACKENDS = ["compile", "triton"]
NPAIRS = 8
REPS = 3
MIN_MEASURE_S = 0.25   # extend reps on very fast cells

PARAMS = dict(tau=0.25, lambda_=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01,
              inner_iterations=30, outer_iterations=10, scale_step=0.8,
              gamma=0.0, median_filtering=5)

# cv2.cuda reference numbers from cv2_gpu.jsonl / the task brief (no median
# filter, real early exit, cheaper bicubic -- see analysis_opencv_cuda.md 5.5).
CV2_CUDA = {(240, 320): (21.1, 9.3), (480, 640): (26.5, 13.1), (720, 1280): (35.8, 21.6)}


def log(*a):
    print(*a, flush=True)


def load_batch(B: int, hw, dev, dtype=torch.float32):
    pairs = load_pairs(max_pairs=NPAIRS, gray=True, size=hw)
    p = torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs])
    n = torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs])
    r = -(-B // len(pairs))
    return (p.repeat(r, 1, 1)[:B].to(device=dev, dtype=dtype),
            n.repeat(r, 1, 1)[:B].to(device=dev, dtype=dtype))


def fresh(backend: str):
    """Drop cached compiles/tuning so each cell tunes at its own B.

    The Triton block config is keyed on image shape only (so results are
    bitwise batch-size independent, see triton_tvl1._pick_config); rebuilding
    the solver per cell is what lets each B pick its own tiling.
    """
    torch_flow._COMPILED_FNS.pop(backend, None)


@torch.no_grad()
def bench_cell(backend: str, p, n) -> dict:
    B = p.shape[0]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fresh(backend)

    def call():
        return torch_flow.calc_flow_tvl1(p, n, backend=backend, early_exit=True, **PARAMS)

    t0 = time.perf_counter()
    out = call()
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0

    # Inductor's compilation/autotuning allocates its own scratch during the
    # warm call, which would land in the peak and make the compile backend look
    # 3-7x hungrier than it is; measure the peak over the timed reps only.
    del out
    torch.cuda.reset_peak_memory_stats()
    ts = []
    while True:
        t0 = time.perf_counter()
        out = call()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        if len(ts) >= REPS and sum(ts) >= MIN_MEASURE_S:
            break
        if len(ts) >= 50:
            break
    t = float(np.median(ts))
    finite = bool(torch.isfinite(out).all())
    return dict(status="ok", warm_s=warm, reps=len(ts), call_s=t,
                ms_per_pair=t / B * 1e3, fps=B / t,
                peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                spread=(max(ts) - min(ts)) / t, finite=finite,
                flow_absmax=float(out.abs().max()))


# --------------------------------------------------------------------------- #
# kernel launches per primal-dual iteration
# --------------------------------------------------------------------------- #
def _launch_counts(prof) -> tuple[int, int]:
    """(cudaLaunchKernel calls, device kernel events) from a finished profile."""
    launches = 0
    for e in prof.key_averages():
        if e.key.startswith(("cudaLaunchKernel", "cuLaunchKernel")):
            launches += e.count
    kernels = 0
    try:
        from torch.autograd import DeviceType
        for e in prof.events():
            if (getattr(e, "device_type", None) == DeviceType.CUDA
                    and getattr(e, "device_index", -1) >= 0):
                kernels += 1
    except Exception:  # noqa: BLE001
        kernels = -1
    return launches, kernels


@torch.no_grad()
def launches_per_iter(backend: str, p, n, its=(10, 30)) -> dict:
    """Launches per iteration = d(launches)/d(inner_iterations).

    One warp, one scale, no median filter, no early exit -- so `inner_iterations`
    is exactly the number of primal-dual iterations and everything else is a
    constant that the difference removes.
    """
    from torch.profiler import ProfilerActivity, profile

    res = {}
    for it in its:
        par = dict(PARAMS)
        par.update(nscales=1, warps=1, outer_iterations=1, inner_iterations=it,
                   median_filtering=1)
        fresh(backend)

        def call(par=par):
            return torch_flow.calc_flow_tvl1(p, n, backend=backend, early_exit=False, **par)

        call()
        torch.cuda.synchronize()   # compile / autotune outside the profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
        res[it] = _launch_counts(prof)
    (i0, i1) = its[0], its[-1]
    (l0, k0), (l1, k1) = res[i0], res[i1]
    return dict(backend=backend, raw={str(k): v for k, v in res.items()},
                launches_per_iter=(l1 - l0) / (i1 - i0),
                kernels_per_iter=((k1 - k0) / (i1 - i0)) if k0 >= 0 else None)


# --------------------------------------------------------------------------- #
# reporting (also usable standalone on an existing JSONL, --summary-only)
# --------------------------------------------------------------------------- #
def load_done(path: str) -> dict[tuple, dict]:
    """Read back the ok cells of a (possibly truncated / resumed) JSONL run.

    A cell measured more than once keeps the *fastest* reading, the usual
    best-of convention: a repeat is only ever slower because of interference
    (another job sharing the node), never faster.
    """
    out: dict[tuple, dict] = {}
    try:
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") == "cell" and r.get("status") == "ok":
                    hw = tuple(int(v) for v in r["res"].split("x"))
                    k = (hw, r["B"], r["backend"], r.get("dtype", "fp32"))
                    if k not in out or r["ms_per_pair"] < out[k]["ms_per_pair"]:
                        out[k] = r
    except FileNotFoundError:
        pass
    return out


def summarize(results: dict[tuple, dict], resolutions, batches, backends):
    def ms(cell):
        return cell["ms_per_pair"] if cell else None

    def cellstr(v, fmt="{:.2f}"):
        return f"{(fmt.format(v) if v is not None else '-'):>9}"

    for dt in ("fp32", "fp16"):
        cells = {k: v for k, v in results.items() if k[3] == dt}
        if not cells:
            continue
        header = "".join(f"{('B=' + str(b)):>9}" for b in batches)
        log(f"\n=== ms/pair, {dt} (lower is better) ===")
        log(f"{'res':>9} {'backend':<9} {header}  {'best':>8}  {'cv2.cuda 1s/4s':>16}")
        for hw in resolutions:
            for be in backends:
                row = [cells.get((hw, b, be, dt)) for b in batches]
                if not any(row):
                    continue
                best = min((r["ms_per_pair"] for r in row if r), default=float("nan"))
                cv = CV2_CUDA.get(hw)
                log(f"{hw[0]}x{hw[1]}".rjust(9) + f" {be:<9} "
                    + "".join(cellstr(ms(r)) for r in row)
                    + f"  {best:>8.2f}  "
                    + (f"{cv[0]:.1f} / {cv[1]:.1f}" if cv else "-").rjust(16))
        log(f"\n=== triton speedup over compile, {dt} ===")
        log(f"{'res':>9} {header}")
        for hw in resolutions:
            row = []
            for b in batches:
                a, c = cells.get((hw, b, "compile", dt)), cells.get((hw, b, "triton", dt))
                row.append(ms(a) / ms(c) if a and c else None)
            if any(v is not None for v in row):
                log(f"{hw[0]}x{hw[1]}".rjust(9) + "".join(cellstr(v, "{:.2f}x") for v in row))
        log(f"\n=== peak MiB, {dt} ===")
        log(f"{'res':>9} {'backend':<9} {header}")
        for hw in resolutions:
            for be in backends:
                row = [cells.get((hw, b, be, dt)) for b in batches]
                if any(row):
                    log(f"{hw[0]}x{hw[1]}".rjust(9) + f" {be:<9} "
                        + "".join(cellstr(r["peak_mib"] if r else None, "{:.0f}") for r in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="triton_grid.jsonl")
    ap.add_argument("--res", default="", help="comma list like 240x320,720x1280")
    ap.add_argument("--batches", default="")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--fp16", action="store_true", help="add fp16 cells at the best B")
    ap.add_argument("--launch-only", action="store_true")
    ap.add_argument("--launch-cell", default="240x320:16")
    ap.add_argument("--skip-launch", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already recorded ok in --out (Inductor warmup "
                         "is ~2 min per new (res, B), so the grid may outlive one srun)")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild the tables from --out without touching the GPU")
    args = ap.parse_args()

    if args.summary_only:
        done = load_done(args.out)
        resolutions = ([tuple(int(v) for v in r.split("x")) for r in args.res.split(",")]
                       if args.res else sorted({k[0] for k in done}, key=lambda t: t[0] * t[1]))
        batches = ([int(v) for v in args.batches.split(",")] if args.batches
                   else sorted({k[1] for k in done}))
        summarize(done, resolutions, batches, args.backends.split(","))
        return

    assert torch.cuda.is_available(), "needs the GPU node"
    torch._dynamo.config.cache_size_limit = 4096
    log(f"# gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    import triton
    log(f"# triton={triton.__version__}")

    resolutions = ([tuple(int(v) for v in r.split("x")) for r in args.res.split(",")]
                   if args.res else RESOLUTIONS)
    batches = ([int(v) for v in args.batches.split(",")] if args.batches else BATCHES)
    backends = args.backends.split(",")
    fh = open(args.out, "a")

    def emit(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    # ---------------- launches per iteration ----------------
    if not args.skip_launch:
        rs, bs = args.launch_cell.split(":")
        hw = tuple(int(v) for v in rs.split("x"))
        p, n = load_batch(int(bs), hw, "cuda")
        log(f"\n== kernel launches per primal-dual iteration ({rs}, B={bs}) ==")
        log(f"{'backend':<10} {'launches/iter':>14} {'kernels/iter':>13}   raw (inner_it: launches, kernels)")
        for be in backends:
            try:
                r = launches_per_iter(be, p, n)
                r.update(kind="launches", res=rs, B=int(bs))
                emit(r)
                log(f"{be:<10} {r['launches_per_iter']:>14.2f} "
                    f"{(r['kernels_per_iter'] if r['kernels_per_iter'] is not None else float('nan')):>13.2f}"
                    f"   {r['raw']}")
            except Exception as exc:  # noqa: BLE001
                log(f"{be:<10} FAILED {type(exc).__name__}: {exc}")
                traceback.print_exc()
        del p, n
        torch.cuda.empty_cache()
        if args.launch_only:
            return

    # ---------------- the grid ----------------
    log(f"\n{'res':>9} {'B':>3} {'backend':<9} {'warm_s':>7} {'ms/pair':>9} {'fps':>8} "
        f"{'peakMiB':>8} {'reps':>4} {'spread':>7}  speedup")
    results: dict[tuple, dict] = load_done(args.out) if args.resume else {}
    if results:
        log(f"# resuming: {len(results)} cells already recorded in {args.out}")
    for hw in resolutions:
        rs = f"{hw[0]}x{hw[1]}"
        oom_from = {be: 10 ** 9 for be in backends}
        for B in batches:
            p = n = None
            for be in backends:
                if B >= oom_from[be]:
                    log(f"{rs:>9} {B:>3} {be:<9} skipped (OOM at smaller B)")
                    continue
                if args.resume and (hw, B, be, "fp32") in results:
                    continue
                try:
                    if p is None:
                        p, n = load_batch(B, hw, "cuda")
                    r = bench_cell(be, p, n)
                except torch.cuda.OutOfMemoryError as exc:
                    oom_from[be] = B
                    torch.cuda.empty_cache()
                    log(f"{rs:>9} {B:>3} {be:<9} OOM")
                    emit(dict(kind="cell", res=rs, B=B, backend=be, dtype="fp32",
                              status="oom", err=str(exc)[:200]))
                    continue
                except Exception as exc:  # noqa: BLE001
                    torch.cuda.empty_cache()
                    log(f"{rs:>9} {B:>3} {be:<9} FAILED {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    emit(dict(kind="cell", res=rs, B=B, backend=be, dtype="fp32",
                              status="fail", err=f"{type(exc).__name__}: {exc}"[:300]))
                    continue
                r.update(kind="cell", res=rs, B=B, backend=be, dtype="fp32",
                         px_per_pair=hw[0] * hw[1])
                results[(hw, B, be, "fp32")] = r
                emit(r)
                base = results.get((hw, B, "compile", "fp32"))
                sp = (f"{base['ms_per_pair'] / r['ms_per_pair']:.2f}x vs compile"
                      if base and be != "compile" else "")
                log(f"{rs:>9} {B:>3} {be:<9} {r['warm_s']:>7.1f} {r['ms_per_pair']:>9.2f} "
                    f"{r['fps']:>8.1f} {r['peak_mib']:>8.0f} {r['reps']:>4} "
                    f"{r['spread']:>7.3f}  {sp}"
                    + ("" if r["finite"] else "  !! NON-FINITE"))
            del p, n
            torch.cuda.empty_cache()

        # -- optional fp16 at the best B for this resolution
        if args.fp16:
            cand = [(v["ms_per_pair"], k[1]) for k, v in results.items()
                    if k[0] == hw and k[2] == "triton" and k[3] == "fp32"]
            if cand:
                bB = min(cand)[1]
                p, n = load_batch(bB, hw, "cuda", dtype=torch.float16)
                for be in backends:
                    try:
                        r = bench_cell(be, p, n)
                    except Exception as exc:  # noqa: BLE001
                        log(f"{rs:>9} {bB:>3} {be:<9} fp16 FAILED {type(exc).__name__}: {exc}")
                        emit(dict(kind="cell", res=rs, B=bB, backend=be, dtype="fp16",
                                  status="fail", err=f"{type(exc).__name__}: {exc}"[:300]))
                        continue
                    r.update(kind="cell", res=rs, B=bB, backend=be, dtype="fp16",
                             px_per_pair=hw[0] * hw[1])
                    results[(hw, bB, be, "fp16")] = r
                    emit(r)
                    b32 = results.get((hw, bB, be, "fp32"))
                    sp = f"{b32['ms_per_pair'] / r['ms_per_pair']:.2f}x vs fp32" if b32 else ""
                    log(f"{rs:>9} {bB:>3} {be + '/f16':<9} {r['warm_s']:>7.1f} "
                        f"{r['ms_per_pair']:>9.2f} {r['fps']:>8.1f} {r['peak_mib']:>8.0f} "
                        f"{r['reps']:>4} {r['spread']:>7.3f}  {sp}"
                        + ("" if r["finite"] else "  !! NON-FINITE"))
                del p, n
                torch.cuda.empty_cache()

    # ---------------- summary tables ----------------
    summarize(results, resolutions, batches, backends)

    step = torch_flow._COMPILED_FNS.get("triton", (None,))[0]
    if step is not None and hasattr(step, "config_report"):
        log("\n=== Triton block configs (last cell's solver) ===")
        for line in step.config_report():
            log("  " + line)
    fh.close()


if __name__ == "__main__":
    main()
