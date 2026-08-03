"""Resolution x batch x backend speedup map with sampled GPU utilization.

Sweeps {tvl1, deepflow} x {eager, compile} x B x {240x320, 480x640, 480x854,
720x1280} on one GPU.  For every timed region an `nvidia-smi ... -lms 200`
subprocess samples utilization.gpu / memory.used, so each cell reports not just
ms/pair but how busy the SM array actually was while producing it.

  srun -p gpu -w ilps-cn119 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
       --time=01:30:00 ./run_nvme.sh python res_speedup.py --out /path/res.jsonl

  ... --mode probe                    # one cheap cell per (algo,res,backend)
  ... --algos tvl1 --res 720x1280     # subset, resumable via --out (appends)

Results are appended as JSON lines to --out *as they finish*, and echoed to
stdout prefixed with "RESULT ", so a job that hits its time limit still leaves
every completed cell behind.  Re-running with the same --out and --resume skips
cells already present in the file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import torch

from benchmark import DEEPFLOW_PARAMS, TVL1_PARAMS
from benchmark_data import load_pairs
from torch_deepflow import calc_flow_deepflow
from torch_flow import calc_flow_tvl1

SIZES: dict[str, tuple[int, int]] = {
    "240x320": (240, 320),
    "480x640": (480, 640),
    "480x854": (480, 854),
    "720x1280": (720, 1280),
}
# batch grid per resolution (memory / wall-clock sensible)
BATCH_PLAN: dict[str, list[int]] = {
    "240x320": [4, 16, 64],
    "480x640": [4, 16, 64],
    "480x854": [4, 16, 32],
    "720x1280": [2, 8, 16],
}

MIN_REPS = 2           # timed reps, min-of-N reported
MAX_REPS = 6
TARGET_REGION_S = 3.0  # keep the timed region long enough for >=10 util samples
CELL_BUDGET_S = 420.0  # skip a cell whose projected timed cost exceeds this


# --------------------------------------------------------------------------- #
# GPU utilization sampling (nvidia-smi subprocess; pynvml is not installed)
# --------------------------------------------------------------------------- #
class GpuSampler:
    """Background `nvidia-smi -lms <interval>` poller for one GPU.

    nvidia-smi reports every GPU visible in the job's device cgroup, which on a
    shared node can be more than ours, so rows are filtered by the GPU UUID
    torch reports for the current device (falling back to "all rows" when the
    UUID is unavailable and only one GPU is visible).
    """

    def __init__(self, interval_ms: int = 200) -> None:
        self.interval_ms = interval_ms
        self.exe = shutil.which("nvidia-smi")
        self.uuid = self._torch_uuid()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.rows: list[tuple[str, int, int]] = []  # (uuid, util%, mem_used MiB)
        self.note = ""

    @staticmethod
    def _torch_uuid() -> str | None:
        try:
            u = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "uuid", None)
            return f"GPU-{u}" if u is not None else None
        except Exception:  # noqa: BLE001 - older torch has no .uuid
            return None

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            uuid, util, mem = parts
            try:
                self.rows.append((uuid, int(util), int(mem)))
            except ValueError:  # "[N/A]" on some driver/vGPU setups
                continue

    def start(self) -> None:
        self.rows = []
        self.note = ""
        if self.exe is None:
            self.note = "nvidia-smi not found"
            return
        cmd = [self.exe,
               "--query-gpu=uuid,utilization.gpu,memory.used",
               "--format=csv,noheader,nounits",
               f"-lms={self.interval_ms}"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True,
                                         bufsize=1)
        except Exception as exc:  # noqa: BLE001
            self.note = f"spawn failed: {exc}"
            self.proc = None
            return
        self.thread = threading.Thread(target=self._reader, args=(self.proc,), daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            if self.thread is not None:
                self.thread.join(timeout=5)
            self.proc = None
        rows = self.rows
        seen = {r[0] for r in rows}
        note = self.note
        if self.uuid is not None and len(seen) > 1:
            mine = [r for r in rows if r[0] == self.uuid]
            if mine:
                rows = mine
            else:  # UUID formatting mismatch -- cannot attribute; flag it
                note = f"uuid {self.uuid} not among {sorted(seen)}"
        # discard the first and last sample: they straddle the timed region
        core = rows[1:-1] if len(rows) >= 3 else rows
        out: dict = {"util_n": len(core), "util_note": note,
                     "util_gpus_seen": len(seen)}
        if not core:
            out.update(util_mean=None, util_max=None, smi_mem_max=None)
            if not note:
                out["util_note"] = f"only {len(rows)} raw samples"
            return out
        u = [r[1] for r in core]
        m = [r[2] for r in core]
        out.update(util_mean=sum(u) / len(u), util_max=max(u), smi_mem_max=max(m))
        return out


# --------------------------------------------------------------------------- #
# Data / backends
# --------------------------------------------------------------------------- #
def make_fn(algo: str, backend: str):
    if algo == "tvl1":
        return lambda p, n: calc_flow_tvl1(p, n, backend=backend, **TVL1_PARAMS)
    return lambda p, n: calc_flow_deepflow(p, n, backend=backend, **DEEPFLOW_PARAMS)


_BASE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def load_batch(B: int, hw: tuple[int, int], dev: str = "cuda"):
    """B real Middlebury pairs on `dev` (4 distinct pairs, tiled to fill B)."""
    if hw not in _BASE:
        pairs = load_pairs(max_pairs=4, gray=True, size=hw)
        if not pairs:
            raise RuntimeError("no Middlebury pairs found")
        _BASE[hw] = (
            torch.stack([torch.from_numpy(a.astype(np.float32)) for a, _, _ in pairs]),
            torch.stack([torch.from_numpy(b.astype(np.float32)) for _, b, _ in pairs]),
        )
    bp, bn = _BASE[hw]
    r = -(-B // bp.shape[0])
    return bp.repeat(r, 1, 1)[:B].to(dev), bn.repeat(r, 1, 1)[:B].to(dev)


def raise_dynamo_limits() -> None:
    """One graph per (pyramid-level shape x batch); the defaults are far too low."""
    import torch._dynamo as dyn

    for name, val in (("cache_size_limit", 8192), ("recompile_limit", 8192),
                      ("accumulated_cache_size_limit", 262144),
                      ("accumulated_recompile_limit", 262144)):
        if hasattr(dyn.config, name):
            try:
                setattr(dyn.config, name, val)
            except Exception:  # noqa: BLE001 - some aliases are read-only
                pass


def clear_gpu() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)


# --------------------------------------------------------------------------- #
# One cell
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_cell(algo: str, backend: str, res: str, B: int, sampler: GpuSampler,
             min_reps: int = MIN_REPS) -> dict:
    """Warm up (compiles), then time reps with GPU utilization sampled throughout."""
    hw = SIZES[res]
    fn = make_fn(algo, backend)
    rec: dict = {"algo": algo, "res": res, "backend": backend, "B": B,
                 "px_per_pair": hw[0] * hw[1]}
    p = n = None
    t_cell = time.perf_counter()
    try:
        p, n = load_batch(B, hw)
        clear_gpu()
        t0 = time.perf_counter()
        fn(p, n)
        torch.cuda.synchronize()
        rec["warmup_s"] = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats()
        sampler.start()
        time.sleep(0.35)  # let the sampler emit its (discarded) first row
        ts: list[float] = []
        t_region = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            fn(p, n)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            enough = len(ts) >= min_reps and (time.perf_counter() - t_region) >= TARGET_REGION_S
            if enough or len(ts) >= MAX_REPS:
                break
        time.sleep(0.35)  # ... and its (discarded) last row
        rec.update(sampler.stop())

        t = min(ts)
        rec.update(reps=len(ts), call_s=t, ms_per_pair=t / B * 1e3, fps=B / t,
                   spread=(max(ts) - t) / t,
                   peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                   status="ok")
    except Exception as exc:  # noqa: BLE001 - OOM/compile failures must not stop the sweep
        sampler.stop()
        rec["status"] = "OOM" if is_oom(exc) else "ERR"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        del p, n
        clear_gpu()
    rec["cell_s"] = time.perf_counter() - t_cell
    return rec


HEADER = (f"{'algo':<9} {'res':>8} {'backend':<8} {'B':>4} {'warm_s':>8} "
          f"{'ms/pair':>9} {'fps':>8} {'util%':>7} {'utmax':>6} {'n':>3} "
          f"{'peakMiB':>8} {'cell_s':>7}")


def fmt_row(r: dict) -> str:
    head = f"{r['algo']:<9} {r['res']:>8} {r['backend']:<8} {r['B']:>4}"
    if r["status"] != "ok":
        return f"{head} {r['status']:>8}  {r.get('error', r.get('reason', ''))}"
    um = r.get("util_mean")
    ux = r.get("util_max")
    return (f"{head} {r['warmup_s']:>8.1f} {r['ms_per_pair']:>9.2f} {r['fps']:>8.1f} "
            f"{(f'{um:.1f}' if um is not None else '-'):>7} "
            f"{(f'{ux}' if ux is not None else '-'):>6} {r.get('util_n', 0):>3} "
            f"{r['peak_mib']:>8.0f} {r['cell_s']:>7.1f}")


class Emitter:
    def __init__(self, path: str | None) -> None:
        self.path = path

    def __call__(self, rec: dict) -> None:
        print(fmt_row(rec), flush=True)
        print("RESULT " + json.dumps(rec), flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()


def load_done(path: str | None) -> set[tuple]:
    done: set[tuple] = set()
    if not path or not os.path.isfile(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add((r["algo"], r["res"], r["backend"], r["B"]))
    return done


# --------------------------------------------------------------------------- #
def sweep(args) -> None:
    emit = Emitter(args.out)
    done = load_done(args.out) if args.resume else set()
    sampler = GpuSampler(args.interval_ms)
    print(f"# gpu={torch.cuda.get_device_name(0)} uuid={sampler.uuid} "
          f"torch={torch.__version__} smi={sampler.exe}", flush=True)
    print(HEADER, flush=True)
    t_start = time.perf_counter()
    for res in args.res:
        batches = args.batches or BATCH_PLAN[res]
        for algo in args.algos:
            for backend in args.backends:
                last = None  # (B, call_s) of the largest successful cell
                dead = None
                for B in batches:
                    key = (algo, res, backend, B)
                    if key in done:
                        print(f"# skip (done) {key}", flush=True)
                        continue
                    if time.perf_counter() - t_start > args.wall_budget_s:
                        emit({**dict(zip(("algo", "res", "backend", "B"), key)),
                              "status": "SKIP", "reason": "wall budget exhausted"})
                        continue
                    if dead:
                        emit({**dict(zip(("algo", "res", "backend", "B"), key)),
                              "status": "SKIP", "reason": dead})
                        continue
                    if last is not None:
                        proj = last[1] * B / last[0] * (MIN_REPS + 1)
                        if proj > CELL_BUDGET_S:
                            emit({**dict(zip(("algo", "res", "backend", "B"), key)),
                                  "status": "SKIP",
                                  "reason": f"projected {proj:.0f}s > {CELL_BUDGET_S:.0f}s"})
                            dead = "budget"
                            continue
                    rec = run_cell(algo, backend, res, B, sampler, args.reps)
                    emit(rec)
                    if rec["status"] == "ok":
                        last = (B, rec["call_s"])
                    elif rec["status"] == "OOM":
                        dead = "OOM at smaller B"
                    else:
                        dead = "error at smaller B"
    print(f"# total {time.perf_counter() - t_start:.0f}s", flush=True)


def probe(args) -> None:
    """One cheap cell per (algo, res, backend): what does compile cost here?"""
    emit = Emitter(args.out)
    sampler = GpuSampler(args.interval_ms)
    print(f"# gpu={torch.cuda.get_device_name(0)} uuid={sampler.uuid} "
          f"smi={sampler.exe}", flush=True)
    print(HEADER, flush=True)
    for res in args.res:
        for algo in args.algos:
            for backend in args.backends:
                emit(run_cell(algo, backend, res, args.probe_batch, sampler, min_reps=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sweep", "probe"], default="sweep")
    ap.add_argument("--algos", nargs="+", default=["tvl1", "deepflow"],
                    choices=["tvl1", "deepflow"])
    ap.add_argument("--backends", nargs="+", default=["eager", "compile"],
                    choices=["eager", "compile", "cudagraphs"])
    ap.add_argument("--res", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--batches", nargs="+", type=int, default=None,
                    help="override the per-resolution batch grid")
    ap.add_argument("--reps", type=int, default=MIN_REPS, help="minimum timed reps")
    ap.add_argument("--probe-batch", type=int, default=2)
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--wall-budget-s", type=float, default=4200.0)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already recorded ok in --out")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("res_speedup.py needs a GPU (run it under srun)")
    torch.backends.cudnn.benchmark = True
    raise_dynamo_limits()
    (probe if args.mode == "probe" else sweep)(args)


if __name__ == "__main__":
    main()
