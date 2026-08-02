# optic_flow — batched PyTorch dense optical flow vs OpenCV

CPU/GPU benchmarks of classical dense optical flow (Dual TV-L1, DeepFlow,
Farneback), plus **pure-PyTorch, batched reimplementations** of TV-L1 and
DeepFlow that match OpenCV's output (EPE vs cv2 ≤ 0.03 px, DeepFlow bitwise)
and run unmodified on CPU or CUDA.

对经典稠密光流（Dual TV-L1 / DeepFlow / Farneback）做 CPU vs GPU 性能对比，
并提供与 OpenCV 数值对齐的纯 PyTorch 批量实现：支持一次并行处理任意多的
帧对、连续帧序列、乃至多组序列，适合 GPU 吞吐场景。

## Layout

| file | what |
|---|---|
| `torch_flow.py` | Dual TV-L1 (Zach et al. 2007) in pure torch — `calc_flow_tvl1(prev, next)` on `(B,H,W)` → `(B,2,H,W)`, plus `calc_flow_tvl1_video` for `(T,H,W)` / `(N,T,H,W)` sequences |
| `torch_deepflow.py` | DeepFlow variational part (Weinzaepfel et al. 2013, = OpenCV's `createOptFlow_DeepFlow`) — `calc_flow_deepflow` / `calc_flow_deepflow_video`, true red-black SOR, bitwise-matches OpenCV |
| `benchmark.py` | unified benchmark: `--algo tvl1\|deepflow\|farneback` × backends `opencv_cpu / opencv_cuda / torch_cpu / torch_cuda`; reports ms/pair, FPS, EPE vs cv2 and vs ground truth |
| `benchmark_data.py` | Middlebury "other" set loader (`load_pairs`, `load_flow_gt`) |
| `download_data.sh` | fetches the Middlebury data (~25 MB) into `data/` |
| `slides/` | bilingual Beamer deck: algorithms, why OpenCV-GPU is slow, batched-torch design, results |
| `slurm_smoke.sh` | GPU smoke test via SLURM |

## Setup

```bash
uv sync                  # numpy, opencv-contrib-python-headless, torch (CUDA wheel)
./download_data.sh       # Middlebury benchmark data
```

`opencv-contrib-python-headless` is deliberate: contrib for `cv2.optflow`
(DualTVL1, DeepFlow), headless so `import cv2` works on GPU compute nodes
without system `libGL`.

## Run

```bash
uv run python benchmark.py                          # TV-L1, all available backends
uv run python benchmark.py --algo deepflow --size 240 320
uv run python benchmark.py --backends opencv_cpu torch_cpu --opencv-threads 1

# on a SLURM GPU node:
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:30:00 \
     uv run --no-sync python benchmark.py --backends torch_cuda
```

Self-tests (compare against OpenCV + ground truth, check batch/video consistency):

```bash
uv run python torch_flow.py
uv run python torch_deepflow.py
```

## Notes

- pip OpenCV wheels ship **without** CUDA, and `cv2.cuda_*` processes one pair
  at a time — the motivation for the batched torch rewrite.
- Torch implementations are slower than OpenCV on CPU (fully vectorized code
  pays off only with GPU parallelism + batching).
- Inputs are grayscale in 0–255 range (matching OpenCV's uint8→float
  convention; DeepFlow's ζ-normalization is not intensity-scale invariant).
