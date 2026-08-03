# OpenCV CUDA optical flow vs. our batched PyTorch reimplementations

**Source-level engineering analysis.** No benchmarks were run for this document; every claim
about OpenCV is traced to a file/line in the sources listed in §0. Our own timings are quoted
from `res_gpu_a.log` / `res_gpu_b.log` / `res_cpu.log` in this repo (NVIDIA L40, 142 SMs).

> **Measured (custom CUDA 12.9 build, L40):** cv2.cuda TVL1 **21.1 ms/pair @ 240x320** single-pair
> resident (42% GPU util), **9.3 ms/pair** with 4 streams x 4 engine instances (92% util);
> 26.5 / 13.1 ms at 480x640. Inside the source-derived 5-20 ms estimate of §3.6 once the 1-3 ms
> texture-object overhead is added. See §5.5 for why these numbers are **not** directly comparable
> to ours at default parameters (no median filter, real early exit, cheaper bicubic).

---

## 0. Sources actually fetched

| # | URL (all `4.x` branch) | Lines | Role |
|---|---|---|---|
| 1 | `opencv_contrib/modules/cudaoptflow/src/tvl1flow.cpp` | 393 | CUDA TV-L1 host driver |
| 2 | `opencv_contrib/modules/cudaoptflow/src/cuda/tvl1flow.cu` | 366 | CUDA TV-L1 kernels |
| 3 | `opencv_contrib/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp` | — | public API + defaults |
| 4 | `opencv_contrib/modules/cudaoptflow/src/farneback.cpp` | 492 | CUDA Farneback host driver |
| 5 | `opencv_contrib/modules/cudaoptflow/src/cuda/farneback.cu` | ~620 | CUDA Farneback kernels |
| 6 | `opencv_contrib/modules/cudaoptflow/src/cuda/pyrlk.cu`, `src/pyrlk.cpp` | — | context (sparse/dense LK) |
| 7 | `opencv_contrib/modules/cudaoptflow/test/test_optflow.cpp` | — | **CPU-vs-CUDA parity test** (§5.5) |
| 8 | `opencv_contrib/modules/optflow/src/tvl1flow.cpp` | 1700+ | **CPU** TV-L1 (the one we ported) |
| 9 | `opencv_contrib/modules/optflow/src/deepflow.cpp` | 6.2 KB | DeepFlow (CPU-only wrapper) |
| 10 | `opencv/modules/video/src/variational_refinement.cpp` | 59 KB | `cv::VariationalRefinement` (CPU-only) |
| 11 | `opencv_contrib/modules/cudev/include/opencv2/cudev/ptr2d/texture.hpp` | — | texture-object defaults |

**Path correction vs. the task brief:** Farneback CUDA does **not** live in the opencv main repo.
`opencv/modules/video/src/optflowgf.cpp` is the CPU/OpenCL implementation; the CUDA one is
`opencv_contrib/modules/cudaoptflow/src/{farneback.cpp,cuda/farneback.cu}`. Directory listing of
`modules/cudaoptflow/src` (GitHub API) returns exactly: `brox.cpp`, `farneback.cpp`,
`nvidiaOpticalFlow.cpp`, `pyrlk.cpp`, `tvl1flow.cpp`, `precomp.hpp`, `cuda/`; and
`cuda/` contains only `farneback.cu`, `nvidiaOpticalFlow.cu`, `pyrlk.cu`, `tvl1flow.cu`.
**There is no CUDA DeepFlow, no CUDA VariationalRefinement, and no CUDA DIS anywhere.**

Our code: `/fnwi_fs/ivi/irlab/personal/tlong/code/optic_flow/torch_flow.py` (722 lines, 544 of
implementation + 178 of self-test) and `torch_deepflow.py` (749 lines).

> **中文小结**：CUDA 版 Farneback 在 `opencv_contrib/modules/cudaoptflow` 而非主仓库；`cudaoptflow`
> 模块一共只有 5 个算法（Brox / Farneback / NVOFA 硬件 / PyrLK / TV-L1），**完全没有 DeepFlow 或
> 变分细化的 CUDA 实现**。

---

## 1. Kernel design of `cv::cuda::OpticalFlowDual_TVL1`

### 1.1 The complete kernel inventory — only **three** custom kernels

`tvl1flow.cu` defines exactly three `__global__` functions (all inside `namespace tvl1flow`):

| Kernel | Line | Launched | Does |
|---|---|---|---|
| `centeredGradientKernel` | `tvl1flow.cu:59` | once per pyramid level | `dx = 0.5*(I1[x+1]-I1[x-1])`, `dy` likewise, clamped borders |
| `warpBackwardKernel` | `tvl1flow.cu:106` | once per warp (5×/level) | bicubic warp of `I1`, `I1x`, `I1y` **+** `grad = Ix²+Iy²` **+** `rho_c` linearization — **4 outputs from one pass** |
| `estimateUKernel` | `tvl1flow.cu:209` | once per iteration | **thresholding (v-step) + divergence + primal update + per-pixel error** — fully fused |
| `estimateDualVariablesKernel` | `tvl1flow.cu:313` | once per iteration | forward gradient of `u` + dual ascent + reprojection, fused |

Everything else is delegated to existing `cudaarithm` / `cudawarping` primitives from the host
side (`tvl1flow.cpp`): `cuda::resize` (pyramid build **and** flow prolongation),
`cuda::multiply` (flow rescale by `1/scaleStep`), `GpuMat::setTo` (zeroing the duals),
`convertTo` (uint8→float ×1.0, or float→float ×255), `cuda::merge` (u1,u2 → CV_32FC2), and
`cuda::calcSum` (the convergence reduction).

**There is no separate warping kernel, no separate gradient kernel, no separate divergence
kernel, and no separate thresholding kernel.** This is the headline: OpenCV hand-fused the
entire primal-dual iteration into **2 kernel launches**.

### 1.2 Fusion depth of `estimateUKernel` (the hot kernel)

Reading `tvl1flow.cu:209-288`, one thread does, in registers:

1. loads `I1wx, I1wy, grad, rho_c, u1, u2` (+`u3` if `gamma`);
2. re-linearizes `rho = rho_c + I1wx*u1 + I1wy*u2 + gamma*u3`;
3. the 3-branch soft-thresholding operator `TH` (`rho < -l_t*grad` / `rho > l_t*grad` / interior
   `fi = -rho/grad`, guarded by `gradVal > numeric_limits<float>::epsilon()`) producing `d1,d2,d3`;
4. `divergence(p11,p12,y,x)` and `divergence(p21,p22,y,x)` — an inlined `__device__` helper
   (`tvl1flow.cu:187`) implementing the backward-difference divergence with the four corner/edge
   special cases;
5. primal update `u1 = v1 + theta*div_p1`, written **in place**;
6. conditionally (`calcError`, a kernel argument) `error(y,x) = (Δu1)² + (Δu2)²`.

In our `_tvl1_step` (`torch_flow.py:146-228`) those same six stages are ~30 separate eager
ATen ops. This is the single biggest structural difference.

`estimateDualVariablesKernel` (`tvl1flow.cu:313-348`) is similarly fused: forward differences of
`u1,u2` (with `::min(x+1, cols-1)` clamping), `::hypotf`, the normalizer `1 + taut*|∇u|`, and the
in-place division of all four `p` planes.

### 1.3 Texture memory — yes, but **not** for filtering

`warpBackward` (`tvl1flow.cu:166-181`) builds three texture objects **per call**:

```cpp
cv::cudev::Texture<float> texI1(I1);
cv::cudev::Texture<float> texI1x(I1x);
cv::cudev::Texture<float> texI1y(I1y);
```

The `cv::cudev::Texture` constructor defaults (`texture.hpp:231-233`) are
`normalizedCoords=false, filterMode=cudaFilterModePoint, addressMode=cudaAddressModeClamp`, and
`TexturePtr::operator()(y,x)` is a plain `tex2D<R>(tex, x, y)` (`texture.hpp:31-33`).

So the texture unit provides (a) the **read-only texture cache** for the 4×4–5×5 warp
neighbourhood and (b) **free `cudaAddressModeClamp` border handling** — but the interpolation is
done in software by `bicubicCoeff` (`tvl1flow.cu:89-104`) over
`[ceil(wx-2), floor(wx+2)] × [ceil(wy-2), floor(wy+2)]`, i.e. 16–25 taps × 3 images = 48–75
texture fetches per pixel, with the weights renormalized by `1/wsum`.

Two consequences worth flagging:

* **This is not the hardware bilinear path.** Contrast `pyrlk.cu`, which does use
  hardware-interpolated fetches. Our `F.grid_sample(..., mode="bicubic")` is also
  software-interpolated (Inductor emits the taps), so on this axis **we are not at a
  disadvantage** — both do ~16 gathers per output.
* **Creating three `cudaTextureObject_t` per `warpBackward` call is a real cost.**
  `cudaCreateTextureObject`/`cudaDestroyTextureObject` are driver calls in the tens of
  microseconds. With 5 scales × 5 warps = 25 calls → 75 create + 75 destroy per solve, plausibly
  **1–3 ms of pure driver overhead per image pair**. This is a regression introduced when OpenCV
  migrated from statically-declared `texture<>` references to texture objects (OpenCV ≥ 4.8);
  the objects could trivially be hoisted to per-level scope.

### 1.4 Shared memory — **none in TV-L1** (but heavy use in Farneback)

`tvl1flow.cu` contains **zero** `__shared__` declarations. Every kernel is a pure
one-thread-one-pixel stencil relying on L1/L2/texture cache for the ±1 neighbour reads. This is
defensible: the stencils are 2-point, so the cache-hit rate is high and a shared-memory tiling
would add a `__syncthreads()` for little gain.

For contrast, `farneback.cu` — the same team, a harder stencil — is aggressively hand-optimized:

* `__constant__ float c_g[8], c_xg[8], c_xxg[8]` and `c_ig11, c_ig03, c_ig33, c_ig55`
  (`farneback.cu:60-63`), `__constant__ float c_border[BORDER_SIZE+1]` (`:154`),
  `__constant__ float c_gKer[MAX_KSIZE_HALF+1]` (`:453`) — filter taps in constant memory;
* `extern __shared__ float smem[]` in `polynomialExpansion` (`:75`), `boxFilter5` (`:364`),
  `gaussianBlur` (`:463`), `gaussianBlur5` (`:547`) — row-tiled separable convolutions;
* a **halo-overlapped 1-D grid** for the polynomial expansion:
  `grid(divUp(src.cols, block.x - 2*polyN), src.rows)` (`farneback.cu:139`) — one block per
  image row-tile with `polyN` overlap on each side.

So OpenCV *does* reach for shared memory when the arithmetic intensity justifies it; TV-L1's
elementwise structure simply doesn't.

### 1.5 Convergence check: a device reduction **plus a host stall**, run sparsely

From `tvl1flow.cpp:352-379`:

```cpp
const double scaledEpsilon = epsilon_ * epsilon_ * I0.size().area();
...
double error = std::numeric_limits<double>::max();
double prevError = 0.0;
for (int n = 0; error > scaledEpsilon && n < iterations_; ++n)
{
    // some tweaks to make sum operation less frequently
    bool calcError = (epsilon_ > 0) && (n & 0x1) && (prevError < scaledEpsilon);
    estimateU(..., calcError, stream);
    if (calcError)
    {
        cuda::calcSum(diff, diff_sum_dev, cv::noArray(), _stream);
        diff_sum_dev.download(diff_sum_host, _stream);
        _stream.waitForCompletion();          // <-- FULL HOST SYNC
        error = diff_sum_host.at<double>(0,0);
        prevError = error;
    }
    else
    {
        error = std::numeric_limits<double>::max();
        prevError -= scaledEpsilon;
    }
    estimateDualVariables(..., stream);
}
```

Three things to note.

1. **`calcError` is a kernel argument**, not a separate kernel — the per-pixel squared update is
   written by `estimateUKernel` itself only when needed, so on non-checking iterations the kernel
   skips both the arithmetic and the `error(y,x)` store.
2. **The reduction is a genuine device-side reduction** (`cuda::calcSum` → `cudaarithm` `sum`),
   accumulated in **`double`** (`diff_sum_host.at<double>(0,0)`), but it is followed by a
   `download` + `_stream.waitForCompletion()` — a **full host/device round-trip that drains the
   whole stream**. This is the one place where the CUDA TV-L1 pipeline bubbles.
3. **The `prevError -= scaledEpsilon` trick** is an elegant adaptive-skip heuristic: it assumes
   the error decreases by roughly one `scaledEpsilon` per iteration, so after measuring
   `error = E` it skips ≈ `E/scaledEpsilon` iterations before checking again. Net effect: only a
   handful of syncs per warp (typically 3–8), not 150.

### 1.6 Kernel-launch budget per solve

Defaults (`cudaoptflow.hpp:375-385`): `tau=0.25, lambda=0.15, theta=0.3, nscales=5, warps=5,
epsilon=0.01, iterations=300, scaleStep=0.8, gamma=0.0`.

Per pyramid level: `1` (`centeredGradient`) + `4` (`setTo` on p11,p12,p21,p22)
+ `warps × (1 warpBackward + iterations × 2)` = `5 + 5×(1 + 600)` = **3 010**.

| Stage | Launches (240×320, worst case = no early exit) |
|---|---|
| 5 levels × 3 010 | 15 050 |
| pyramid: 2 `convertTo` + 8 `resize` + 2 `setTo` | 12 |
| prolongation: 4 levels × (2 `resize` + 2 `multiply`) | 16 |
| `cuda::merge` | 1 |
| **Total (upper bound)** | **≈ 15 080** |
| + a handful of `calcSum` reductions & host syncs | +10–50 |

**Head-to-head, at the identical 7 500 primal-dual iterations** (OpenCV `iterations=300` per warp
== our `outer_iterations=10 × inner_iterations=30`; 5 scales × 5 warps × 300 = 7 500 both):

| Implementation | Launches / solve | Launches / primal-dual iteration | Ratio vs. OpenCV |
|---|---|---|---|
| `cv2.cuda` TV-L1 (1 pair) | **≈ 15.1 k** | **2** | 1× |
| ours, `backend="eager"` | ≈ 595 k (whole batch) | ≈ 79 | **39×** |
| ours, `backend="compile"` | ≈ 52 k (whole batch) | ≈ 6.9 | **3.5×** |

But per *image pair* the picture inverts, because our 52 k covers the whole batch:

| B | `cv2.cuda` launches for B pairs | our compiled launches for B pairs | our advantage |
|---|---|---|---|
| 1 | 15 k | 52 k | 0.29× (worse) |
| 4 | 60 k | 52 k | 1.15× |
| 16 | 241 k | 52 k | **4.6×** |
| 64 | 965 k | 52 k | **18.5×** |

The break-even batch size is **B ≈ 3.5**. Below that OpenCV issues fewer launches than we do;
above it, batching wins outright.

> **中文小结**：cv2.cuda TV-L1 只有 **3 个自定义 kernel**，把整个原始-对偶迭代手工融合成
> **每次迭代 2 次 launch**（阈值化+散度+primal 更新在 `estimateUKernel` 一个 kernel 内完成）。
> 纹理内存只用于 **只读缓存 + 边界 clamp**（`cudaFilterModePoint`），双三次插值仍是软件实现，
> 因此在这一点上我们的 `grid_sample` 并不吃亏。TV-L1 **完全没有用 shared memory**（Farneback 则大量使用
> constant/shared）。收敛检查是真正的 device 端 double 精度归约，但每次检查都要 `waitForCompletion()`
> 全流同步；OpenCV 用 `prevError -= scaledEpsilon` 的自适应跳过技巧把同步次数压到每 warp 几次。
> 默认参数下一次求解约 **15 080 次 kernel launch**（单张图），我们 eager 约 595k、compile 约 52k
> （整个 batch）；按每对图算，**B≈3.5 是盈亏平衡点**，B=16 时我们少 4.6 倍，B=64 时少 18.5 倍。

---

## 2. Parallelization model

### 2.1 One thread per pixel, fixed 32×8 blocks, everywhere

All three TV-L1 kernels open with the identical preamble:

```cpp
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= src.cols || y >= src.rows) return;
```

and all three launch with a **hard-coded, non-tuned** configuration:

```cpp
const dim3 block(32, 8);                                   // 256 threads
const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
```

(`tvl1flow.cu:73-74`, `:128-129`, `:298-299`, `:355-356`.) There is no occupancy calculator, no
device-capability branch, no persistent-block / grid-stride formulation, no vectorized (`float4`)
load path. 32 threads along `x` gives coalesced 128-byte transactions per warp — that is the
whole optimization.

### 2.2 There is **no batch axis anywhere** — confirmed

Traced end to end:

* Public API: `virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow, Stream&)`
  (`tvl1flow.cpp:123`) → `_frame0.getGpuMat()` — a single 2-D `GpuMat`.
* `CV_Assert( I0.type() == CV_8UC1 || I0.type() == CV_32FC1 )` (`tvl1flow.cpp:186`) — strictly
  single-channel 2-D. No `CV_32FC(N)`, no 3-D `GpuMat` (OpenCV's `GpuMat` has no N-D support at all).
* All 15 scratch buffers (`I1x_buf … diff_buf`, `tvl1flow.cpp:145-166`) are **2-D class members**,
  sub-`Rect`-ed per level: `GpuMat I1x = I1x_buf(Rect(0, 0, I0.cols, I0.rows));`
* All kernels index `PtrStepSzf` — a `{float* data; size_t step; int rows, cols;}` 2-D view. A
  batch dimension is not expressible in that type.

**Concurrency workaround OpenCV itself recommends** is visible in their own test
(`test_optflow.cpp:468-527`, `TVL1AsyncParallelLoopBody`): `NUM_STREAMS` **separate algorithm
instances**, each `create()`d inside the parallel loop body, each on its own `cv::cuda::Stream`.
That means N× the scratch memory, N× the host threads, and *still* one kernel launch per image per
iteration — it hides launch latency but does nothing for per-kernel occupancy. It also confirms
that a single `OpticalFlowDual_TVL1` object is **not** reusable across threads/streams, since the
buffers are mutable members.

### 2.3 Occupancy arithmetic on an L40 (142 SMs)

Pyramid at 240×320 with `scaleStep=0.8` (`cvRound` chain, `>=16 px` cutoff): five levels
240×320 → 192×256 → 154×205 → 123×164 → 98×131.

| Level | Size | Grid `(divUp(w,32), divUp(h,8))` | Blocks | Blocks / 142 SMs |
|---|---|---|---|---|
| 0 | 240×320 | (10, 30) | **300** | 2.11 waves |
| 1 | 192×256 | (8, 24) | **192** | 1.35 waves |
| 2 | 154×205 | (7, 20) | **140** | **0.99** — one block/SM, *no* latency hiding |
| 3 | 123×164 | (6, 16) | **96** | **0.68** — 46 SMs idle |
| 4 | 98×131 | (5, 13) | **65** | **0.46** — 77 SMs idle |

Since every level runs the same `warps × iterations = 1 500` iterations, **60 % of the solve
(levels 2–4) executes with at most one 256-thread block per SM** — i.e. ≤ 256/1536 ≈ **17 %
theoretical occupancy**, and on levels 3–4 a majority of the GPU is literally idle. Even the
finest level tops out at 2 blocks/SM = 512/1536 ≈ **33 %**, with a 0.11-wave tail.

At these sizes the kernels are also **latency-bound, not bandwidth-bound**:
`estimateUKernel` moves ~14 planes × 76 800 px × 4 B ≈ 4.3 MB at level 0 → ~6 µs at L40's
~700 GB/s, but only ~0.7 µs of traffic at level 4 — below the ~2–3 µs floor of a kernel launch.
Levels 3–4 are pure launch overhead.

**This is exactly the gap our `(B, C, H, W)` layout closes.** Our `_tvl1_step` operates on
`(B, 2, h, w)`, so the same level-4 stencil that gives OpenCV 65 blocks gives us `65 × B` blocks
of equivalent work: at B=16 that is 1 040 → 7.3 waves, fully saturating the L40. Our measured
GPU utilization tracks this precisely (`res_gpu_a.log`, TV-L1 240×320 compiled):

| B | util | ms/pair |
|---|---|---|
| 4 | 47.4 % | 119.4 |
| 16 | **85.1 %** | **54.5** |
| 64 | 96.9 % | 73.2 (memory pressure, 1 582 MiB) |

> **中文小结**：三个 kernel 全部是"一像素一线程"，block 固定 `(32,8)=256` 线程，没有任何 occupancy
> 调优。`calc()` 只接受单张 2-D `GpuMat`（`CV_8UC1`/`CV_32FC1`），`PtrStepSzf` 类型本身就无法表达 batch
> 维度——**全链路确认没有 batch 轴**。在 240×320 的 5 层金字塔上，第 2/3/4 层分别只有 140/96/65 个
> block，而 L40 有 142 个 SM：**60% 的求解时间里 GPU 至多每 SM 一个 block**（≈17% occupancy），
> 最粗两层甚至一半 SM 空转。OpenCV 官方的并发方案是"N 个算法实例 + N 个 Stream"，只能掩盖 launch
> 延迟，无法提升单 kernel 的 occupancy。我们的 `(B,2,h,w)` 布局正好把这个洞补上：B=16 时利用率 85%。

---

## 3. Where OpenCV's approach is genuinely stronger

### 3.1 Hand fusion beats autofusion — by ~3.5×, even against `torch.compile`

2 launches/iteration vs. our compiled ~6.9. Inductor cannot fuse across the whole `_tvl1_step`
because of (a) `torch.cat` materializations (`torch_flow.py:181, 190-193, 209-217`), (b) the
`err = ((u_new-u)**2).sum(dim=(1,2,3))` reduction in the middle of the body
(`torch_flow.py:195`) which forces a fusion break, and (c) the shifted-slice `F.pad` patterns in
`_divergence` / `_forward_gradient`. A hand-written kernel keeps `v`, `div_p`, `u_new` entirely
in registers; Inductor round-trips several of them through HBM.

Concretely: OpenCV's `estimateUKernel` reads 6 planes and writes 2–3. Our compiled equivalent
reads/writes closer to 20 planes across ~7 kernels — roughly **2.5–3× the DRAM traffic per
iteration**.

### 3.2 In-place updates and a ~3.5× smaller memory footprint

`estimateUKernel` writes `u1(y,x)` / `u2(y,x)` **in place** while reading them, and
`estimateDualVariablesKernel` writes `p11..p22` in place. This is safe precisely *because* they
are two separate kernels: the primal kernel touches `u` only at its own pixel (the neighbour reads
are on `p`), and the dual kernel touches `p` only at its own pixel (the neighbour reads are on
`u`). Splitting the iteration at exactly that boundary is a deliberate, elegant design choice —
it buys in-place safety at the cost of one extra launch.

Footprint per pair at 240×320 fp32:

| | OpenCV CUDA | ours (compiled, measured) |
|---|---|---|
| pyramid `I0s+I1s` (5 levels, 190 532 px total) | 1.52 MB | — |
| pyramid `u1s+u2s` | 1.52 MB | — |
| 12 full-res scratch planes (`I1x…diff`) | 3.69 MB | — |
| **total** | **≈ 6.7 MB** | **24.7 MiB** (395.8 MiB / B=16) |

**≈ 3.7× more memory per pair on our side**, which is what caps our batch: at 480×640 B=64 we hit
6.3 GiB and the ms/pair *regresses* (367 vs 323 at B=16, `res_gpu_a.log`).

### 3.3 Zero Python / dispatch overhead

Our eager path spends ~79 ATen dispatches per iteration × 7 500 iterations = ~595 k trips through
the PyTorch dispatcher, autograd-key checks, `TensorImpl` allocation, and the CUDA caching
allocator. That is the entire explanation for eager's 172 ms/pair even at B=64 with 96.8 %
reported utilization — the "utilization" is `nvidia-smi` sampling, not achieved FLOPs.
OpenCV's host loop is a `for` over `cudaLaunchKernel` with a pre-packed `PtrStepSzf` struct:
~1–2 µs each, no allocation, no dispatch.

### 3.4 Real early exit vs. our freeze mask

OpenCV's `for (int n = 0; error > scaledEpsilon && n < iterations_; ++n)` **actually terminates**.
With `epsilon=0.01` at 240×320, `scaledEpsilon = 1e-4 × 76 800 = 7.68`; in practice TV-L1 at the
coarse levels converges in tens of iterations, not 300. A 3–10× reduction in real iteration count
is typical.

Ours (`torch_flow.py:337, 199, 218, 227`) keeps a per-sample `active` mask and runs the **full**
fixed trip count, merely freezing converged samples with `torch.where`. The GPU work is unchanged
— we pay 300 iterations always. This was a deliberate torch.compile-safety decision (no
data-dependent control flow, no graph breaks, static shapes for CUDA graphs), but it is a
**real, large, unrecovered cost**: plausibly the single biggest remaining speedup available to us.

Mitigation worth considering: a host-side check every K iterations on `active.any()` (one
`.item()` sync per K iterations, exactly OpenCV's tradeoff) would break out of the Python loop when
the *whole batch* has converged. With batching, though, the batch converges only as slowly as its
slowest member — so the benefit shrinks as B grows. That is a genuine, unavoidable batching tax.

### 3.5 Multi-stream pipelining (in Farneback, not TV-L1)

`farneback.cpp` uses `Stream streams[5]` (`:319`), an `Event sourceStreamComplete` (`:143`), and
`streams[1].waitEvent(sourceStreamComplete)` (`:344`) to overlap the two frames' `convertTo` and
`pyrDown` chains. TV-L1 uses a single stream throughout — the algorithm is strictly serial, so
there is nothing to overlap.

### 3.6 What their runtime probably looks like (source-derived estimate)

Summing `iterations × 2 × max(traffic_time, ~2.5 µs launch floor)` over the five levels at
240×320, the **no-early-exit** upper bound is ≈ 48 ms/pair (level 0 ≈ 16 ms, level 1 ≈ 10 ms,
levels 2–4 ≈ 7.5 ms each, all launch-floor-bound). With the default `epsilon=0.01` early exit
cutting the effective iteration count by 3–10×, the expected range is **≈ 5–20 ms/pair**, plus
1–3 ms of `cudaCreateTextureObject` overhead (§1.3).

> Measured: **21.1 ms/pair** (resident, B=1) / **9.3 ms/pair** (4 streams), 240x320 — the
> single-pair number lands at the top of the 5-20 ms estimate, consistent with the early exit
> being partially offset by the per-warp texture-object churn.

> **中文小结**：OpenCV 的优势是真实的。(1) 手工融合 2 launch/迭代，比我们 compile 后的 ~6.9 还少
> 3.5 倍，DRAM 流量约为我们的 1/2.5~1/3；(2) 原地更新 + 15 个复用 buffer，每对图只用 ~6.7 MB，
> 是我们 24.7 MiB 的 1/3.7；(3) 没有 Python dispatch 开销；(4) **epsilon 提前退出是真退出**，
> 而我们跑满固定 300 次迭代只用 mask 冻结——这是我们目前最大的一块未回收的性能损失。

---

## 4. Where our approach is genuinely stronger

### 4.1 Batching — structurally impossible for them, and it is the whole ballgame at small sizes

Everything in §2.3. Their design cannot amortize small images: `PtrStepSzf` is 2-D, `GpuMat` is
2-D, the buffers are 2-D members. Adding a batch axis to `cudaoptflow` would mean changing the
pointer type, the buffer management, all three kernels' index math, and the public `calc()`
signature — i.e. a new module, not a patch.

For the video use case this is decisive: `calc_flow_tvl1_video` (`torch_flow.py:494-543`) turns a
`(T,H,W)` clip into a single `(T-1)`-wide batch with zero copies (`frames[:-1], frames[1:]` are
views), and processes all pairs in one solve. OpenCV must loop.

### 4.2 `torch.compile` gets within 3.5× of hand-fusion for free

52 k vs 15 k launches, from a 544-line Python file with no CUDA code. Measured effect
(`res_gpu_a.log`, 240×320, B=16): **248.4 → 54.5 ms/pair, a 4.6× speedup**, purely from Inductor
fusing the ~79 eager kernels of `_tvl1_step` down to ~7. The `cudagraphs` backend
(`torch_flow.py:241-274`) removes the remaining launch overhead entirely by replaying the
iteration as a captured graph — an option OpenCV's stream-based C++ loop does not have without a
rewrite.

### 4.3 dtype flexibility — theirs is **fp32-only, verified**

Verified in source:

* `PtrStepSzf` = `PtrStepSz<float>` — every kernel signature in `tvl1flow.cu` uses it exclusively;
  there are no templates over `T` and no `__half` / `nv_bfloat16` includes.
* `CV_Assert( I0.type() == CV_8UC1 || I0.type() == CV_32FC1 )` (`tvl1flow.cpp:186`); the uint8 path
  is immediately widened: `I0.convertTo(I0s[0], CV_32F, ...)` (`tvl1flow.cpp:200`).
* Every buffer is `create(..., CV_32FC1)` (`tvl1flow.cpp:159-181`).
* The `double`-typed parameters (`tau_, lambda_, theta_`) are all `static_cast<float>` at the
  launch site (`tvl1flow.cpp:344-345`).

There is no fp16/bf16/TF32 path, and no way to add one without templating all three kernels and
the entire host driver. Our solver is dtype-polymorphic by construction: it inherits the input
dtype and every intermediate is created with the working dtype — hence `bf16_study.py` exists at
all. On a bandwidth/launch-bound solver, fp16 halves the traffic of every one of those ~20 planes
per iteration (measured: TV-L1 1.75×, DeepFlow 2.10× at B=64).

### 4.4 DeepFlow: **no CUDA implementation exists in OpenCV at all** — verified

* `modules/cudaoptflow/src/` contains only Brox, Farneback, NvidiaOpticalFlow (the NVOFA hardware
  block), PyrLK, TV-L1. No DeepFlow, no variational refinement.
* `opencv_contrib/modules/optflow/src/deepflow.cpp` (`class OpticalFlowDeepFlow : public
  DenseOpticalFlow`) has a CPU-only `void calc(InputArray, InputArray, InputOutputArray)` — note
  the signature has **no `Stream&`**, which is OpenCV's marker for a CPU-only algorithm.
* `opencv/modules/video/src/variational_refinement.cpp` (59 KB) — grepping for
  `cuda|opencl|ocl_|UMat` returns **zero** matches. The only parallelism is
  `cv::parallel_for_` over row stripes (`:1168-1185`) plus `CV_SIMD128` `v_float32x4` intrinsics
  (`:636, 843, 955, 1059`). `modules/optflow/src/opencl/` contains only
  `optical_flow_tvl1.cl`, `sparse_matching_gpc.cl`, `updatemotionhistory.cl` — no DeepFlow kernel.

**So `torch_deepflow.py` has no OpenCV GPU counterpart whatsoever.** Our measured 16.0 ms/pair
(compiled, B=64, 240×320) vs. OpenCV CPU's 297.4 ms/pair (48 threads, `res_cpu.log`) is an
**18.6× speedup over the only implementation that exists**. At B=16 it is 24.3 ms/pair (12.2×).

This also makes the porting-effort argument concrete: the SOR solver is genuinely awkward for
CUDA (Gauss-Seidel is sequential), and OpenCV never did it. We got it by expressing red-black SOR
as two masked simultaneous updates (`torch_deepflow.py:213-235`) — 23 lines.

### 4.5 Composability and maintainability

* **Composability**: our output is a `torch.Tensor` on-device, feedable straight into a
  downstream network (two-stream action recognition, flow-guided video models) with no
  `download()`/`upload()`. `cv2.cuda` returns a `GpuMat`; interop with torch requires either a
  host round-trip or a fragile `__cuda_array_interface__` dance.
* **LOC / build surface**: OpenCV's TV-L1 is `tvl1flow.cpp` (393) + `tvl1flow.cu` (366) +
  ~90 lines of abstract interface in `cudaoptflow.hpp` + CMake + the `HAVE_CUDA` /
  `CUDA_DISABLER` dispatch stubs (`tvl1flow.cpp:44-52`) — **≈ 850 lines across 3 files, 2
  languages, requiring nvcc and a rebuild of a 100+ MB module to change one line**. Ours is
  **544 lines of implementation in one file**, editable and rerunnable in seconds.
* **Statefulness**: OpenCV's 15 scratch buffers are mutable class members → not thread-safe, and
  `nscales_ = s;` at `tvl1flow.cpp:236` **mutates the object's own configuration** when the pyramid
  is truncated, so `getNumScales()` silently returns a different value after the first small image
  and every subsequent call uses the reduced pyramid. (The same bug is in the CPU version,
  `optflow/src/tvl1flow.cpp:484`.) Our `calc_flow_tvl1` is a pure function.
* **Extensibility**: adding the `medianFiltering` step (present in CPU TV-L1, absent from CUDA —
  §5.1) is 2 lines for us; for OpenCV it means a new `.cu` kernel, a new host wrapper, a new
  parameter on the abstract interface, and an ABI break.

> **中文小结**：我们的优势同样是结构性的。(1) **batching**——他们的 `PtrStepSzf`/`GpuMat` 是二维类型，
> 加 batch 轴等于重写模块；小图上这决定一切。(2) `torch.compile` 白拿 4.6× 加速（248→54.5 ms/pair），
> 已经逼近手工融合的 3.5 倍以内。(3) **他们是纯 fp32**（源码已核实：`PtrStepSzf`、`CV_32FC1`、
> `static_cast<float>`，无任何半精度路径），我们的 dtype 随输入走，fp16 直接减半带宽。
> (4) **OpenCV 根本没有 DeepFlow / 变分细化的任何 GPU 实现**（`variational_refinement.cpp` 里
> `cuda|opencl|UMat` 零命中，只有 `parallel_for_` + SIMD），我们 16.0 ms/pair 对比其 CPU 297.4 ms/pair
> 是 **18.6×**。(5) 544 行单文件 vs 850 行跨 3 文件 2 语言 + nvcc 重编译；且他们的 buffer 是可变成员，
> `nscales_ = s` 还会永久改写对象配置。

---

## 5. Convergence and numerics: the CUDA TV-L1 is a *different algorithm* from the CPU TV-L1

This is the most under-appreciated finding, and it invalidates naive ms/pair comparisons.

### 5.1 No median filtering on the CUDA path

CPU (`optflow/src/tvl1flow.cpp:398`): `medianFiltering = 5` by default, applied at the top of every
outer iteration:

```cpp
for (int n_outer = 0; error > scaledEpsilon && n_outer < outerIterations; ++n_outer) {
    if (medianFiltering > 1) { medianBlur(u1,u1,medianFiltering); medianBlur(u2,u2,medianFiltering); }
    for (int n_inner = 0; error > scaledEpsilon && n_inner < innerIterations; ++n_inner) { ... }
}
```

CUDA: **the parameter does not exist.** There is no `medianFiltering` in
`cuda::OpticalFlowDual_TVL1` (`cudaoptflow.hpp:311-385`), no median kernel in `tvl1flow.cu`, and
no `medianBlur` call in `tvl1flow.cpp`. Median filtering of the flow is the main outlier-remover
in TV-L1 (cf. Sun et al., *Secrets of Optical Flow Estimation*), so **the CUDA version is
measurably less accurate at defaults** and also does strictly less work — at 240×320 it skips
5 scales × 5 warps × 10 = **250 median filters** that both the CPU version and ours perform.

### 5.2 `iterations` semantics: flat 300 vs. our 10 × 30

CUDA has one `iterations` (default **300**, `cudaoptflow.hpp:382`) and one flat loop. CPU/ours have
`outerIterations=10 × innerIterations=30`. The **product is identical (300)**, so the primal-dual
iteration counts match exactly — the only difference is where the median filter is injected. This
makes launch-count comparison (§1.6) exactly apples-to-apples, while making *runtime* comparison
apples-to-oranges (§5.5).

### 5.3 A different bicubic kernel — CUDA ≠ CPU ≠ ours

| | interpolant | border | normalization |
|---|---|---|---|
| CUDA `warpBackwardKernel` | Catmull-Rom, **a = −0.5** (`bicubicCoeff`: `x²(1.5x − 2.5) + 1`) | `cudaAddressModeClamp` (replicate) | divides by `wsum` |
| CPU `remap(..., INTER_CUBIC)` (`optflow/src/tvl1flow.cpp:1372-1374`) | **a = −0.75** | `BORDER_CONSTANT` (0) | none |
| ours `F.grid_sample(mode="bicubic", padding_mode="zeros")` | **a = −0.75** | zeros | none |

Solving `(a+2)|x|³ − (a+3)|x|² + 1` against `1.5x³ − 2.5x² + 1` gives `a = −0.5`; the `1<|x|<2`
branch `−0.5x³ + 2.5x² − 4x + 2` confirms it. **Ours matches the CPU reference; the CUDA version
does not.** The CUDA version also renormalizes by `wsum` — mathematically a no-op for an exact
4-tap partition of unity, but not for the 5-tap case that `ceil(wx−2)…floor(wx+2)` produces when
`wx` is integral, so it is a real (small) difference.

Also note CUDA fuses all three warps (`I1`, `I1x`, `I1y`) into one kernel; CPU issues three
separate `remap` calls; we fuse all three into **one** `grid_sample` by stacking them as channels
— same trick as the CUDA version.

### 5.4 Accumulation precision

| | error reduction | accumulator |
|---|---|---|
| CUDA | `cuda::calcSum` → device tree reduction, `diff_sum_host.at<double>(0,0)` | **double** |
| CPU | accumulated inside `estimateU`'s return value | **float** |
| ours | `((u_new-u)**2).sum(dim=(1,2,3))` | **working dtype** (fp32, or fp16/bf16 in reduced-precision mode) |

Note the amusing inversion: the *GPU* path has the most accurate convergence metric. All three do
the actual solve arithmetic in fp32 (CUDA: registers; CPU: `Mat_<float>`; ours: fp32 tensors).
Our reduced-precision mode is the only configuration that degrades the solve itself, and the error
reduction is the most sensitive part (it sums `H*W` small positive numbers) — worth accumulating
in fp32 even in fp16/bf16 mode.

### 5.5 Epsilon semantics, and why runtime comparison needs a caveat

| | when checked | granularity | on convergence |
|---|---|---|---|
| CUDA | odd `n`, throttled by `prevError -= scaledEpsilon` (≈ every `error/scaledEpsilon` iters) | whole image | **loop breaks** |
| CPU | **every** inner and outer iteration | whole image | **loop breaks** |
| ours | every iteration | **per batch sample** | sample frozen; loop runs to 300 |

Ours is the only one with per-sample granularity (necessary for batching, and strictly *more*
faithful than the CPU when a batch is heterogeneous), and the only one that keeps burning GPU
cycles afterwards.

**Does OpenCV document CPU/CUDA equivalence? No — and their own test proves the opposite.**
`cudaoptflow/test/test_optflow.cpp:440-465` cannot compare the two at default settings. It must
first *cripple the CPU version to match the GPU one*:

```cpp
d_alg->setNumIterations(10);                       // 300 -> 10
...
alg->setMedianFiltering(1);                        // disable the median filter
alg->setInnerIterations(1);
alg->setOuterIterations(d_alg->getNumIterations());
...
EXPECT_MAT_SIMILAR(flow, d_flow, 4e-3);            // similarity, not equality
```

So OpenCV upstream implicitly acknowledges the CUDA TV-L1 is a *variant*, comparable to the CPU
one only with median filtering off and at 10 iterations, and only to a 4e-3 relative tolerance.

**Consequence for benchmarking.** At defaults, `cv2.cuda` TV-L1 performs:
* the same 7 500 primal-dual iterations *at most*, but typically 3–10× fewer thanks to real early
  exit (§3.4);
* **zero** of the 250 median filters that our solver and the CPU solver perform;
* a cheaper (a = −0.5, clamp) warp.

A raw `cv2.cuda ms/pair` vs `torch ms/pair` table would therefore overstate their advantage. The
fair comparisons are:
1. **ours with `median_filtering=1`** vs. `cv2.cuda` at defaults (matches the algorithm, still
   differs on early exit and the bicubic kernel);
2. **ours at defaults** vs. **`cv2` CPU at defaults** (`419.4 ms/pair` at 240×320, 48 threads) —
   this one *is* apples-to-apples, and gives us 419.4 / 54.5 = **7.7× at B=16**, 419.4 / 73.2 =
   5.7× at B=64.

> **中文小结**：这是最容易被忽略的一点——**CUDA 版 TV-L1 和 CPU 版不是同一个算法**。
> (1) CUDA 版**没有中值滤波**（CPU 默认 5×5，每个 outer 迭代一次；240×320 下共 250 次），
> 而中值滤波恰是 TV-L1 去外点的主力，所以 CUDA 版默认精度更低、做的功也更少。
> (2) `iterations=300` 与我们的 `10×30=300` 乘积相同，迭代次数完全可比。
> (3) **双三次核不同**：CUDA 是 a=−0.5 + clamp 边界 + `wsum` 归一化；CPU 与我们都是 a=−0.75 + 零边界
> ——我们对齐的是 CPU 参考实现。
> (4) 收敛量 CUDA 用 **double** 归约，CPU 用 float，我们用工作精度（半精度模式下建议改为 fp32 累加）。
> (5) OpenCV 自己的测试 (`test_optflow.cpp:440`) 必须把 CPU 版的中值滤波关掉、迭代降到 10 次，
> 才能以 4e-3 的相对容差通过——**官方从未声称 CPU/CUDA 等价**。因此直接比 ms/pair 会高估他们的优势；
> 公平的对照是「我们默认 vs cv2 CPU 默认」= 419.4 / 54.5 = **7.7×**。

---

## 6. Verdict table

| Axis | `cv2.cuda` TV-L1 | ours, `backend="eager"` | ours, `backend="compile"` |
|---|---|---|---|
| **Batching** | ❌ none — `GpuMat`/`PtrStepSzf` are 2-D; workaround = N instances × N streams | ✅ `(B,2,H,W)`, one solve for the whole batch | ✅ same |
| **Kernel fusion** | ✅✅ hand-fused, **2 launches/iteration**; primal = threshold+divergence+update+error in one kernel | ❌ ~79 ATen kernels/iteration | ✅ Inductor → **~6.9/iteration** (3.5× OpenCV) |
| **Launches / solve** | ≈ 15.1 k **per pair** | ≈ 595 k **per batch** | ≈ 52 k **per batch** (break-even at B≈3.5) |
| **Texture / warping** | ✅ 3 texture objects, `cudaFilterModePoint` + `AddressModeClamp` → read-only cache + free border; **software** bicubic (a = −0.5). ⚠️ recreates textures every warp call (75 create+destroy/solve) | `grid_sample` bicubic (a = −0.75, matches CPU ref), 3 planes fused into one call | same, Inductor-generated gather |
| **Shared / constant memory** | TV-L1: none. Farneback: `__constant__` taps + `extern __shared__` tiling + halo grids | n/a (Inductor chooses) | n/a |
| **Early exit** | ✅ real `break`; device `double` reduction, adaptively throttled; costs one full `waitForCompletion()` per check | ❌ fixed 300 iters, per-sample freeze mask | ❌ same (deliberate: keeps graphs static / CUDA-graph-capturable) |
| **Convergence granularity** | whole image | ✅ per batch sample | ✅ per batch sample |
| **dtype support** | ❌ **fp32 only** (`PtrStepSzf`, `CV_32FC1`, `static_cast<float>`); input `CV_8UC1`/`CV_32FC1` | ✅ fp32 / fp16 / bf16 / fp64, follows input | ✅ same |
| **Memory / pair (240×320)** | ✅ ≈ 6.7 MB, 15 reused buffers, in-place primal & dual updates | 24.7 MiB | 24.7 MiB |
| **Median filtering (accuracy)** | ❌ absent (CPU has it, default 5) | ✅ present, cv2-CPU-faithful | ✅ present |
| **Algorithms covered** | Brox, Farneback, NVOFA (hardware), PyrLK, TV-L1. **No DeepFlow / variational / DIS on GPU** | TV-L1 + **DeepFlow** (16.0 ms/pair vs OpenCV CPU 297.4 → **18.6×**) | same |
| **CUDA-graph replay** | ❌ not available | — | ✅ `backend="cudagraphs"` |
| **Composability** | `GpuMat`; torch interop needs a copy or `__cuda_array_interface__` | ✅ native `torch.Tensor`, stays on device | ✅ same |
| **Statefulness / thread-safety** | ❌ 15 mutable member buffers; `nscales_ = s` mutates config permanently (`tvl1flow.cpp:236`) | ✅ pure function | ✅ pure (module-level compile cache only) |
| **LOC / build** | ≈ 850 lines, 3 files, C++ + CUDA, nvcc + module rebuild | **544 lines, 1 file** | same |
| **Extensibility** | new kernel + host wrapper + interface change + ABI break | ✅ edit a Python function | ✅ same (recompile is automatic) |
| **Best measured (240×320)** | **21.1 ms/pair** resident B=1 (42 % util); **9.3 ms/pair** with 4 streams (92 % util) — lighter algorithm variant, see §5.5 | 172.3 ms/pair @ B=64 | **54.5 ms/pair @ B=16** (85 % util) |

### Bottom line

**Per single image pair, OpenCV's CUDA TV-L1 should win**, and its engineering deserves the
credit: three kernels, two launches per iteration, in-place updates, 6.7 MB of scratch, a real
early exit, and a `double`-accumulated device-side convergence test. Our compiled solver is within
~3.5× on launch count and ~3× on DRAM traffic — which is a strong showing for autofusion, but not
a win.

**The comparison flips on three axes that OpenCV's architecture cannot follow:**
1. **Throughput at B > 4.** Their 2-D type system forecloses batching, and at 240×320 three of
   five pyramid levels don't even fill an L40's 142 SMs. This is not a tuning gap; it is a design
   ceiling.
2. **dtype.** fp32-only, verified in source. On a launch/bandwidth-bound solver, fp16 is free
   performance we can take and they cannot (measured 1.75–2.1× at B=64).
3. **Coverage.** DeepFlow has *no* GPU implementation in OpenCV — CPU-only `parallel_for_` + SIMD.
   Our 18.6× there is not a speedup over a GPU baseline; it is the only GPU implementation.

And the accuracy caveat cuts our way: their CUDA path silently drops the median filter and uses a
different bicubic kernel, so it is neither the CPU algorithm nor as accurate as it. Ours is a
faithful port of the CPU reference (verified to < 0.5 px EPE in `torch_flow.py`'s self-test).

> **中文总结**：**单张图上 OpenCV CUDA 应该更快**，其工程质量值得肯定：3 个 kernel、每迭代 2 次
> launch、原地更新、6.7 MB 显存、真正的提前退出、double 精度的 device 端收敛归约。我们编译后在
> launch 数上差 3.5 倍、显存流量差约 3 倍——对于"自动融合"来说已经很接近了，但确实没赢。
> **但在三个维度上他们的架构无法跟进**：(1) **B>4 的吞吐**——二维类型系统从根上封死了 batching，
> 且 240×320 下 5 层金字塔有 3 层填不满 L40 的 142 个 SM；(2) **数据类型**——纯 fp32，源码已核实；
> (3) **算法覆盖**——DeepFlow 在 OpenCV 里根本没有 GPU 版本，我们的 18.6× 不是"加速比"，
> 而是"唯一的 GPU 实现"。此外精度上的差异也对我们有利：他们的 CUDA 路径默默去掉了中值滤波、
> 换了双三次核，既不等同于 CPU 版也不如它准；我们是对 CPU 参考实现的忠实移植（自测 EPE < 0.5 px）。

---

## Appendix: reproducing the arithmetic

**Pyramid (240×320, `scaleStep=0.8`, `cvRound`, stop below 16 px):**
`240×320 → 192×256 → 154×205 → 123×164 → 98×131` (5 levels, areas summing to 190 532 px).

**Grid sizes** with `block(32,8)`, `grid(divUp(cols,32), divUp(rows,8))`:
`(10,30)=300`, `(8,24)=192`, `(7,20)=140`, `(6,16)=96`, `(5,13)=65` blocks.

**Launches per level:** `1 (centeredGradient) + 4 (setTo) + warps × (1 + iterations × 2)`
`= 1 + 4 + 5 × (1 + 300 × 2) = 3 010`; × 5 levels + 29 host-primitive launches ≈ **15 080**.

**Iteration parity:** OpenCV `nscales × warps × iterations = 5 × 5 × 300 = 7 500`;
ours `nscales × warps × outer × inner = 5 × 5 × 10 × 30 = 7 500`.
Hence eager `595 000 / 7 500 ≈ 79` and compiled `52 000 / 7 500 ≈ 6.9` launches per iteration,
against OpenCV's exactly 2.

**OpenCV memory (240×320, fp32):** pyramid `I0s+I1s` `190 532 × 2 × 4 B = 1.52 MB`;
`u1s+u2s` `1.52 MB`; 12 full-res scratch planes `12 × 76 800 × 4 B = 3.69 MB`; **≈ 6.7 MB**.
Ours: `395.8 MiB / 16 = 24.7 MiB/pair` (`res_gpu_a.log`).

**Our measured numbers used above** (L40; `res_gpu_a.log`, `res_cpu.log`), 240×320:

| algo | backend | B | ms/pair | util | peak |
|---|---|---|---|---|---|
| tvl1 | eager | 4 / 16 / 64 | 975.1 / 248.4 / 172.3 | 36 / 62 / 97 % | 99 / 395 / 1582 MiB |
| tvl1 | compile | 4 / 16 / 64 | 119.4 / **54.5** / 73.2 | 47 / 85 / 97 % | 99 / 396 / 1582 MiB |
| deepflow | compile | 4 / 16 / 64 | 96.0 / 24.3 / **16.0** | 16 / 27 / 73 % | 78 / 309 / 1233 MiB |
| tvl1 | cv2 CPU (48 thr) | 1 | 419.4 | — | — |
| deepflow | cv2 CPU (48 thr) | 1 | 297.4 | — | — |
