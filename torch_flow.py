"""Batched dense optical flow (Dual TV-L1) in pure PyTorch.

A vectorized, batch-parallel reimplementation of OpenCV's
``cv2.optflow.DualTVL1OpticalFlow`` (opencv_contrib
``modules/optflow/src/tvl1flow.cpp``, based on the IPOL reference
implementation of Sanchez Perez, Meinhardt-Llopis and Facciolo of the method
of Zach, Pock and Bischof, "A Duality Based Approach for Realtime TV-L1
Optical Flow", 2007).

The method minimizes ``E(u) = int lambda*|rho(u)| + |grad u1| + |grad u2|``
where ``rho(u) = I1(x+u0) + grad I1(x+u0) . (u-u0) - I0`` is the linearized
brightness-constancy residual around the current warp point ``u0``.  An
auxiliary field ``v`` splits the data term from the TV term with quadratic
coupling ``|u-v|^2 / (2*theta)``; the algorithm alternates

1. a **closed-form thresholding step** for ``v`` (pixelwise shrinkage of the
   residual ``rho`` against ``lambda*theta*|grad I1w|^2``),
2. **Chambolle's dual projection algorithm** for the TV proximal step:
   ``u = v + theta * div p`` with the dual field updated as
   ``p <- (p + (tau/theta) * grad u) / (1 + (tau/theta) * |grad u|)``,

inside a coarse-to-fine image pyramid (factor ``scale_step``) with ``warps``
re-linearizations per level.  Like OpenCV, the flow is median-filtered
(5x5 by default, via a fixed min/max selection network) at the start of every
outer iteration, and iterations stop early once the summed squared update
falls below ``epsilon^2 * H * W``.  Convergence is tracked per batch sample by
a *freeze mask* -- a converged element stops changing exactly where OpenCV
would have stopped iterating -- and, with ``early_exit=True`` (the default),
the Python iteration loop additionally breaks once *every* sample is frozen,
detected with OpenCV's throttled host-sync schedule (roughly
``error/(epsilon^2 H W)`` iterations skipped between syncs).  Because the
break only fires when the mask has already frozen the whole batch, it changes
runtime and nothing else: results are bit-identical either way.

Everything is written with ``torch`` + ``torch.nn.functional`` only; a batch
of B image pairs is processed in parallel by every operation (convolutions /
grid_sample / elementwise) with no per-pixel Python loops, and the code runs
unmodified on CPU or CUDA.  The only Python loops are the pyramid / warp /
iteration loops, whose bodies are fixed-shape and branch-free, keeping the
structure torch.compile-safe (``early_exit=False`` additionally makes the trip
counts static, as CUDA-graph capture requires).

Input convention
----------------
``prev`` / ``next`` are grayscale images of shape ``(B, H, W)`` in the
**0-255 intensity range** (what cv2 uses internally: 8-bit input is taken
as-is, float input is assumed [0, 1] and scaled by 255).  ``lambda_`` weighs
the data term against the TV term and is tuned for 0-255 data, so pass 0-255;
no rescaling is performed.  The returned flow has shape ``(B, 2, H, W)``:
channel 0 the horizontal (x/u) and channel 1 the vertical (y/v) displacement
in pixels, cv2 sign convention: ``prev[y, x] ~ next[y + v, x + u]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["calc_flow_tvl1", "calc_flow_tvl1_video"]

# std::numeric_limits<float>::epsilon() -- threshold used by OpenCV to decide
# whether |grad I1w|^2 is large enough for the interior shrinkage branch.
_FLT_EPS = 1.1920929e-07
# OpenCV stops adding pyramid levels once a side would drop below 16 px.
_MIN_SIZE = 16


# --------------------------------------------------------------------------- #
# Differential operators (exact OpenCV border semantics)
# --------------------------------------------------------------------------- #
def _centered_gradient(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences with replicated borders. img: (B, C, H, W).

    Interior: ``0.5*(f[i+1]-f[i-1])``; at the image edge this degenerates to
    ``0.5*(f[edge+1]-f[edge])`` -- identical to OpenCV's centeredGradient.
    """
    px = F.pad(img, (1, 1, 0, 0), mode="replicate")
    dx = 0.5 * (px[..., 2:] - px[..., :-2])
    py = F.pad(img, (0, 0, 1, 1), mode="replicate")
    dy = 0.5 * (py[..., 2:, :] - py[..., :-2, :])
    return dx, dy


def _forward_gradient(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences, zero on the last column / row (OpenCV semantics)."""
    dx = F.pad(u[..., 1:] - u[..., :-1], (0, 1, 0, 0))
    dy = F.pad(u[..., 1:, :] - u[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _divergence(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Backward-difference divergence, adjoint of ``_forward_gradient``.

    ``div(y, x) = p1(y, x) - p1(y, x-1) + p2(y, x) - p2(y-1, x)`` with the
    out-of-range terms taken as 0 (OpenCV's divergence).
    """
    d1 = p1 - F.pad(p1[..., :-1], (1, 0, 0, 0))
    d2 = p2 - F.pad(p2[..., :-1, :], (0, 0, 1, 0))
    return d1 + d2


def _median_blur_sort(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Reference median filter: unfold + ``torch.median`` (see `_median_blur`).

    Materializes the ``k*k``-wide neighbourhood tensor and runs a selection
    reduction over it; kept as the correctness reference for the min/max
    network below (and used for kernels larger than 5x5).
    """
    r = ksize // 2
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    patches = xp.unfold(2, ksize, 1).unfold(3, ksize, 1)  # (B, C, H, W, k, k)
    return patches.reshape(*x.shape, ksize * ksize).median(dim=-1).values


# --------------------------------------------------------------------------- #
# Median filter as a fixed min/max selection network
#
# `torch.median` over an unfolded k*k axis is a data-dependent selection kernel
# on a k*k-times-larger tensor -- on GPU it dominated the solver (~47 % of the
# time at 240x320).  A *selection network* computes the same value with a
# straight-line sequence of `minimum`/`maximum` ops on the k*k shifted views:
# nothing is materialized, and torch.compile fuses the whole thing (25 gathers
# + 202 register ops + 1 store for 5x5) into one kernel.
# --------------------------------------------------------------------------- #
def _odd_even_merge_comparators(m: int) -> list[tuple[int, int]]:
    """Batcher's odd-even mergesort network on ``m`` (a power of two) wires."""
    ops: list[tuple[int, int]] = []
    p = 1
    while p < m:
        k = p
        while k >= 1:
            j = k % p
            while j <= m - 1 - k:
                for i in range(min(k, m - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        ops.append((i + j, i + j + k))
                j += 2 * k
            k //= 2
        p *= 2
    return ops


def _selection_network(n: int, k: int) -> tuple[list[tuple[int, int, bool, int]], int, int]:
    """Straight-line min/max program selecting the k-th smallest of n values.

    Built by (1) taking Batcher's odd-even mergesort on the next power of two
    (comparators touching a padding wire are no-ops against ``+inf`` and are
    dropped), (2) dead-code-eliminating everything the median output does not
    depend on, and (3) allocating the surviving values to a minimal register
    file.  Returns ``(program, n_registers, out_register)``; each program entry
    is ``(a, b, is_min, out)`` meaning ``reg[out] = min|max(reg[a], reg[b])``.

    Registers ``0..n-1`` hold the inputs on entry.  Correctness follows from
    the 0-1 principle (verified exhaustively over all 2**25 inputs for 5x5 and
    all 2**9 for 3x3, and against ``torch.median`` in the self-test).
    """
    m = 1 << (n - 1).bit_length()
    comps = [(i, j) for (i, j) in _odd_even_merge_comparators(m) if i < n and j < n]

    # Symbolic execution: every comparator defines two SSA values (min, max).
    wire = list(range(n))
    defs: list[tuple[int, int, bool]] = []  # value id n+t -> (a, b, is_min)
    for i, j in comps:
        a, b = wire[i], wire[j]
        wire[i] = n + len(defs); defs.append((a, b, True))
        wire[j] = n + len(defs); defs.append((a, b, False))
    target = wire[k]

    live: set[int] = set()
    stack = [target]
    while stack:
        x = stack.pop()
        if x < n or x in live:
            continue
        live.add(x)
        a, b, _ = defs[x - n]
        stack += [a, b]
    order = sorted(live)  # SSA ids are topologically ordered by construction

    last_use: dict[int, int] = {}
    for pos, vid in enumerate(order):
        a, b, _ = defs[vid - n]
        last_use[a] = last_use[b] = pos

    slot = {i: i for i in range(n)}
    free = [i for i in range(n) if i not in last_use and i != target]
    nreg = n
    prog: list[tuple[int, int, bool, int]] = []
    for pos, vid in enumerate(order):
        a, b, is_min = defs[vid - n]
        sa, sb = slot[a], slot[b]
        dead = [s for s, v in ((sa, a), (sb, b)) if last_use.get(v) == pos]
        dead = list(dict.fromkeys(dead))
        if dead:
            out, extra = dead[0], dead[1:]
            free.extend(extra)
        elif free:
            out = free.pop()
        else:
            out, nreg = nreg, nreg + 1
        slot[vid] = out
        prog.append((sa, sb, is_min, out))
    return prog, nreg, slot[target]


_NETWORKS: dict[int, tuple[list[tuple[int, int, bool, int]], int, int]] = {}
# Above this many taps the pruned network stops paying for itself; cv2's
# medianBlur only supports 3x3 / 5x5 on CV_32F data anyway.
_MEDIAN_NET_MAX = 25


def _median_blur(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Channelwise ksize x ksize median filter with replicated borders.

    Same semantics (and, up to ties which are exact here, the same values) as
    ``cv2.medianBlur`` on CV_32F data / `_median_blur_sort`, but evaluated with
    a fixed min/max selection network over the k*k shifted views of the padded
    input -- pure elementwise ops, fully fusable, nothing materialized.
    """
    n = ksize * ksize
    if n > _MEDIAN_NET_MAX:
        return _median_blur_sort(x, ksize)
    r = ksize // 2
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    h, w = x.shape[-2], x.shape[-1]
    if n not in _NETWORKS:
        _NETWORKS[n] = _selection_network(n, n // 2)
    prog, nreg, out = _NETWORKS[n]
    reg: list[torch.Tensor | None] = [xp[..., i : i + h, j : j + w] for i in range(ksize) for j in range(ksize)]
    reg += [None] * (nreg - n)
    for a, b, is_min, o in prog:
        reg[o] = torch.minimum(reg[a], reg[b]) if is_min else torch.maximum(reg[a], reg[b])
    return reg[out]


def _median_step(u: torch.Tensor, active: torch.Tensor, ksize: int) -> torch.Tensor:
    """Median-filter the flow of the still-active batch samples (fusable)."""
    return torch.where(active, _median_blur(u, ksize), u)


# --------------------------------------------------------------------------- #
# Compilable hot functions
#
# The two functions below are pure (tensor args in, tensors out, no Python
# data-dependent branching -- the float/None args are static and simply get
# baked into the trace) so they can be wrapped with torch.compile while the
# outer pyramid / warp / iteration loops stay eager.  `_tvl1_step` is the hot
# spot: it runs up to warps * outer * inner (~1500) times per pyramid level.
# --------------------------------------------------------------------------- #
def _tvl1_warp_setup(
    I1g: torch.Tensor,
    I0: torch.Tensor,
    u: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    sx: float,
    sy: float,
    l_t: float,
):
    """Per-warp setup: warp I1 (+gradients) to x+u and linearize the residual.

    grid_sample(align_corners=True, mode="bicubic") matches
    cv2.remap(..., INTER_CUBIC) with the default constant-0 border: both use
    the alpha = -0.75 cubic kernel with zero-valued out-of-image taps.
    Returns (I1wxy, rho_c, thr, pos, safe_grad) with ``I1wxy`` the warped
    gradient as a single (B, 2, h, w) tensor laid out like ``u``, so the whole
    inner iteration broadcasts over the flow-component axis instead of
    concatenating per-component results.
    """
    grid = torch.stack([sx * (xs + u[:, 0]) - 1.0, sy * (ys + u[:, 1]) - 1.0], dim=-1)
    I1wg = F.grid_sample(I1g, grid, mode="bicubic", padding_mode="zeros", align_corners=True)
    I1w, I1wx, I1wy = I1wg[:, 0:1], I1wg[:, 1:2], I1wg[:, 2:3]

    grad = I1wx * I1wx + I1wy * I1wy
    # Constant part of the linearized residual rho(u) around u0 = u.
    rho_c = I1w - I1wx * u[:, 0:1] - I1wy * u[:, 1:2] - I0
    pos = grad > _FLT_EPS
    safe_grad = torch.where(pos, grad, torch.ones_like(grad))
    thr = l_t * grad
    return I1wg[:, 1:3].contiguous(), rho_c, thr, pos, safe_grad


def _tvl1_step(
    u: torch.Tensor,
    u3: torch.Tensor | None,
    px: torch.Tensor,
    py: torch.Tensor,
    p3x: torch.Tensor | None,
    p3y: torch.Tensor | None,
    active: torch.Tensor,
    rho_c: torch.Tensor,
    I1wxy: torch.Tensor,
    thr: torch.Tensor,
    pos: torch.Tensor,
    safe_grad: torch.Tensor,
    l_t: float,
    theta: float,
    taut: float,
    gamma: float,
    scaled_eps: float,
    calc_error: bool,
):
    """One primal-dual iteration (OpenCV inner-loop body); pure function.

    Returns the updated ``(u, u3, px, py, p3x, p3y, active, err_max)``.
    ``active`` freezes converged batch samples (see `_proc_one_scale`).

    The dual field is carried as two ``(B, 2, h, w)`` tensors -- ``px`` = the
    x-derivative components ``(p11, p21)``, ``py`` = the y ones ``(p12, p22)``
    -- rather than one ``(B, 4, h, w)`` block, so that every stage broadcasts
    over the flow-component axis and no ``torch.cat`` is needed inside the hot
    loop (OpenCV's `estimateUKernel` keeps the same values in registers).

    ``calc_error`` is a *static* flag (each value gets its own compiled graph,
    exactly like OpenCV's ``calcError`` kernel argument): when False the
    convergence reduction -- a whole-tensor sum in the middle of an otherwise
    elementwise body, i.e. a hard fusion break -- is skipped entirely and the
    freeze mask is carried over unchanged.
    """
    use_gamma = gamma != 0.0  # static: specialized at trace time

    # ---- thresholding step (proximal operator of the L1 data term):
    # v = u + shrink(rho) * grad I1w
    rho = rho_c + I1wxy[:, 0:1] * u[:, 0:1] + I1wxy[:, 1:2] * u[:, 1:2]
    if use_gamma:
        rho = rho + gamma * u3
    fi = -rho / safe_grad
    lo = rho < -thr
    hi = rho > thr
    d = torch.where(lo, l_t * I1wxy, torch.where(hi, -l_t * I1wxy, torch.where(pos, fi * I1wxy, 0.0)))
    v = u + d
    if use_gamma:
        d3 = torch.where(
            lo, torch.full_like(rho, l_t * gamma),
            torch.where(hi, torch.full_like(rho, -l_t * gamma), torch.where(pos, fi * gamma, 0.0)),
        )
        v3 = u3 + d3

    # ---- primal update: u = v + theta * div p
    u_new = v + theta * _divergence(px, py)
    if use_gamma:
        u3_new = v3 + theta * _divergence(p3x, p3y)
    if calc_error:
        err = ((u_new - u) ** 2).sum(dim=(1, 2, 3))
        if use_gamma:
            err = err + ((u3_new - u3) ** 2).sum(dim=(1, 2, 3))
        # Frozen samples keep producing (discarded) updates; zero their error so
        # the batch maximum reflects the still-active samples only.
        err = torch.where(active.view(-1), err, torch.zeros_like(err))
        err_max = err.max()
    else:
        err_max = None
    u = torch.where(active, u_new, u)
    if use_gamma:
        u3 = torch.where(active, u3_new, u3)

    # ---- dual ascent + reprojection:
    # p = (p + taut * grad u) / (1 + taut * |grad u|)
    ux, uy = _forward_gradient(u)
    ng = 1.0 + taut * torch.hypot(ux, uy)
    px = torch.where(active, (px + taut * ux) / ng, px)
    py = torch.where(active, (py + taut * uy) / ng, py)
    if use_gamma:
        u3x, u3y = _forward_gradient(u3)
        ng3 = 1.0 + taut * torch.hypot(u3x, u3y)
        p3x = torch.where(active, (p3x + taut * u3x) / ng3, p3x)
        p3y = torch.where(active, (p3y + taut * u3y) / ng3, p3y)

    # Deactivate converged samples *after* this iteration's dual update
    # (OpenCV checks the loop condition on entry).
    if calc_error:
        active = active & (err > scaled_eps).view(-1, 1, 1, 1)
    return u, u3, px, py, p3x, p3y, active, err_max


# Lazily-created compiled variants of the hot functions, keyed by backend.
# dynamic=False: each pyramid-level shape compiles once and is then reused
# across warps / iterations / calls (a handful of shapes in total).
_COMPILED_FNS: dict[str, tuple] = {}

# Ablation hook, not part of the public API: set to False to keep the throttled
# `calc_error` schedule (and the per-sample freeze mask it refreshes) but never
# break out of the iteration loop, which isolates the cost of the convergence
# reduction from the cost of the iterations early exit saves.
_BREAK_ON_CONVERGENCE = True

# OpenCV runs the convergence reduction only on a sparse schedule (`calcError`
# is a kernel argument, tvl1flow.cu:209); `_tvl1_step` supports the same via its
# static `calc_error` flag, which halves the number of graphs that contain the
# fusion-breaking whole-tensor sum.  It is *off* by default: skipping the
# reduction also delays the per-sample freeze mask by up to one iteration, and
# an iteration of post-convergence drift is worth ~0.007 px of EPE against the
# cv2 reference (measured on Middlebury 'Dimetrodon'), which is more than the
# ~2 % of runtime it saves.  Set True to trade that accuracy for throughput.
_SPARSE_CHECKS = False


_BACKENDS = ("eager", "compile", "cudagraphs", "triton")


def _resolve_backend(backend: str, device: torch.device) -> tuple:
    """Return (step_fn, warp_fn, median_fn) for `backend`, caching compiles."""
    if backend == "eager":
        return _tvl1_step, _tvl1_warp_setup, _median_step
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend in ("cudagraphs", "triton") and device.type != "cuda":
        raise RuntimeError(f"backend={backend!r} requires CUDA input tensors")
    if backend not in _COMPILED_FNS:
        import torch._dynamo

        # One graph per (backend, pyramid-level shape, calc_error); make sure
        # dynamo never hits its recompile limit and silently falls back to eager.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        mode = "reduce-overhead" if backend == "cudagraphs" else None
        warp = torch.compile(_tvl1_warp_setup, dynamic=False, mode=mode)
        median = torch.compile(_median_step, dynamic=False, mode=mode)
        if backend == "triton":
            # Only the 2-kernel primal/dual pair is hand-written; the per-warp
            # setup (grid_sample) and the median network stay Inductor-compiled.
            from triton_tvl1 import TritonTVL1Step

            step = TritonTVL1Step()
        else:
            step = torch.compile(_tvl1_step, dynamic=False, mode=mode)
        if backend == "cudagraphs":
            step, warp, median = _cudagraph_safe(step), _cudagraph_safe(warp), _cudagraph_safe(median)
        _COMPILED_FNS[backend] = (step, warp, median)
    return _COMPILED_FNS[backend]


def _err_scalar(err) -> float:
    """Host-side batch-maximum error (the one throttled sync per check window).

    ``_tvl1_step`` already reduces to a 0-dim tensor; the Triton step returns the
    per-sample ``(B,)`` vector instead, because folding the batch axis on-device
    would cost a third kernel launch on every iteration while the value is only
    ever needed on the rare iterations that actually sync.
    """
    return float(err.max()) if err.ndim else float(err)


def _cudagraph_safe(fn):
    """Make a reduce-overhead-compiled fn safe for feed-back loops.

    CUDA-graph outputs live in static buffers that the next replay
    overwrites; we mark each call as a new step and clone the outputs so
    results that escape the iteration loop (e.g. the final flow of a
    pyramid level) stay valid.
    """
    def wrapped(*args):
        torch.compiler.cudagraph_mark_step_begin()
        out = fn(*args)
        if isinstance(out, tuple):
            return tuple(o.clone() if isinstance(o, torch.Tensor) else o for o in out)
        return out.clone()
    return wrapped


# --------------------------------------------------------------------------- #
# Single pyramid level: warps x (outer x inner) primal-dual iterations
# --------------------------------------------------------------------------- #
def _proc_one_scale(
    I0: torch.Tensor,
    I1: torch.Tensor,
    u: torch.Tensor,
    u3: torch.Tensor | None,
    *,
    tau: float,
    lambda_: float,
    theta: float,
    warps: int,
    epsilon: float,
    inner_iterations: int,
    outer_iterations: int,
    gamma: float,
    median_filtering: int,
    early_exit: bool = False,
    step_fn=_tvl1_step,
    warp_fn=_tvl1_warp_setup,
    median_fn=_median_step,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the TV-L1 solver at one pyramid level.

    I0, I1: (B, 1, h, w) images; u: (B, 2, h, w) flow (updated in place of the
    return value); u3: (B, 1, h, w) illumination multiplier field (gamma != 0
    only).  Mirrors OpticalFlowDual_TVL1::procOneScale.  ``step_fn`` /
    ``warp_fn`` / ``median_fn`` are `_tvl1_step` / `_tvl1_warp_setup` /
    `_median_step` or compiled variants.
    """
    b, _, h, w = I0.shape
    dt, dev = I0.dtype, I0.device
    use_gamma = gamma != 0.0

    l_t = lambda_ * theta
    taut = tau / theta
    scaled_eps = float(epsilon) * float(epsilon) * h * w
    early_exit = early_exit and epsilon > 0.0  # cf. OpenCV's `epsilon_ > 0` guard

    # Gradient of the *unwarped* target image (computed once per level).
    I1x, I1y = _centered_gradient(I1)
    I1g = torch.cat([I1, I1x, I1y], dim=1)  # warp all three in one call

    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=dt, device=dev),
        torch.arange(w, dtype=dt, device=dev),
        indexing="ij",
    )
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)

    # Dual variables are reset once per pyramid level (as in OpenCV).
    px = torch.zeros(b, 2, h, w, dtype=dt, device=dev)  # (p11, p21)
    py = torch.zeros(b, 2, h, w, dtype=dt, device=dev)  # (p12, p22)
    p3x = torch.zeros(b, 1, h, w, dtype=dt, device=dev) if use_gamma else None
    p3y = torch.zeros(b, 1, h, w, dtype=dt, device=dev) if use_gamma else None

    for _warp in range(warps):
        # Warp I1 and its gradient to x + u0 (bicubic, zero border) and
        # linearize the residual around the current flow.
        I1wxy, rho_c, thr, pos, safe_grad = warp_fn(I1g, I0, u, xs, ys, sx, sy, l_t)

        # Per-sample convergence mask emulating OpenCV's epsilon early stop:
        # once a sample's summed squared update drops below eps^2*h*w its
        # state (u, p, median filtering) is frozen for the rest of this warp.
        active = torch.ones(b, 1, 1, 1, dtype=torch.bool, device=dev)

        # Convergence bookkeeping, in two tiers:
        #
        # * the error *reduction* (and with it the freeze mask) runs on a
        #   deterministic schedule -- every iteration, or OpenCV's `n & 0x1`
        #   under `_SPARSE_CHECKS` -- never one that depends on a measured
        #   value, so when a sample freezes depends only on that sample and not
        #   on what else is in the batch; results stay bit-identical to the
        #   per-pair solve (`calc_flow_tvl1_video` relies on this);
        # * the *host sync* that can break the loop is throttled exactly like
        #   OpenCV (tvl1flow.cpp:352-379): after measuring `error`, spend one
        #   scaled_eps per iteration before looking again, i.e. skip about
        #   error/scaled_eps iterations between syncs.  Throttling this tier is
        #   free of accuracy risk: the loop is only left once *every* sample is
        #   frozen, at which point the remaining iterations are no-ops -- so
        #   early exit is pure speed, bit-for-bit.
        n_it, prev_err, done = 0, 0.0, False

        for _outer in range(outer_iterations):
            if median_filtering > 1:
                u = median_fn(u, active, median_filtering)
                # The median filter perturbs u, so the "error decreases by one
                # scaled_eps per iteration" estimate the throttle rides on is
                # stale; re-measure.  (OpenCV's CUDA path has no median filter
                # and so never hits this.)  Sync frequency only -- no effect on
                # the result.
                prev_err = 0.0
            for _inner in range(inner_iterations):
                check = ((n_it & 1) == 1) if (early_exit and _SPARSE_CHECKS) else True
                u, u3, px, py, p3x, p3y, active, err_max = step_fn(
                    u, u3, px, py, p3x, p3y, active,
                    rho_c, I1wxy, thr, pos, safe_grad,
                    l_t, theta, taut, gamma, scaled_eps, check,
                )
                n_it += 1
                if early_exit:
                    if check and prev_err < scaled_eps:
                        prev_err = _err_scalar(err_max)  # the one host sync
                        if prev_err <= scaled_eps and _BREAK_ON_CONVERGENCE:
                            done = True  # whole batch converged
                            break
                    else:
                        prev_err -= scaled_eps
            if done:
                break

    return u, u3


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def calc_flow_tvl1(
    prev: torch.Tensor,
    next: torch.Tensor,  # noqa: A002 - name mandated by the cv2-like API
    *,
    tau: float = 0.25,
    lambda_: float = 0.15,
    theta: float = 0.3,
    nscales: int = 5,
    warps: int = 5,
    epsilon: float = 0.01,
    inner_iterations: int = 30,
    outer_iterations: int = 10,
    scale_step: float = 0.8,
    gamma: float = 0.0,
    median_filtering: int = 5,
    use_initial_flow: bool = False,
    initial_flow: torch.Tensor | None = None,
    backend: str = "eager",
    early_exit: bool = True,
) -> torch.Tensor:
    """Batched dense optical flow with the Dual TV-L1 method.

    Parameter names and defaults match ``cv2.optflow.DualTVL1OpticalFlow``.

    Args:
        prev, next: grayscale images, shape ``(B, H, W)``, intensity range
            0-255 (see module docstring).  Float tensors on any device
            (integer tensors are converted to float32).  A batch of B pairs
            is processed in parallel.
        tau: dual ascent time step (stability for tau <= 0.25).
        lambda_: data-term weight; smaller = smoother flow (tuned for 0-255).
        theta: coupling ("tightness") between u and the auxiliary field v.
        nscales: number of pyramid scales (clamped so every level keeps both
            sides >= 16 px, like cv2).
        warps: relinearizations (warpings of ``next``) per scale.
        epsilon: stopping criterion: iterations for a batch sample freeze once
            its summed squared flow update < ``epsilon^2 * H * W``.
        inner_iterations: primal-dual iterations per outer iteration.
        outer_iterations: outer iterations (each starts with a median filter).
        scale_step: pyramid downscale factor per level (in (0, 1)).
        gamma: weight of an additional illumination-gain term (0 disables it,
            as in cv2).
        median_filtering: kernel of the median filter applied to the flow at
            each outer iteration (values <= 1 disable it).  Extra keyword
            relative to the minimal API; matches cv2's default of 5.
        use_initial_flow: seed the pyramid with ``initial_flow`` instead of 0.
        initial_flow: ``(B, 2, H, W)`` seed flow, required when
            ``use_initial_flow`` is True.
        backend: execution backend for the hot per-iteration update
            (`_tvl1_step`) and the per-warp setup (`_tvl1_warp_setup`):

            - ``"eager"`` (default): plain PyTorch, byte-identical to the
              reference implementation.
            - ``"compile"``: ``torch.compile(..., dynamic=False)`` -- fuses
              the ~30 elementwise kernels of each iteration; each pyramid
              level shape compiles once (lazily, cached at module level and
              reused across calls).  Note: on CPU, Inductor needs a C++20
              compiler (GCC >= 10); point the ``CXX`` env var at one if the
              system default is older.
            - ``"cudagraphs"``: ``torch.compile(..., mode="reduce-overhead",
              dynamic=False)`` -- additionally replays CUDA graphs to remove
              launch overhead (the solver is launch-bound on GPU).  Requires
              CUDA input tensors.
            - ``"triton"``: the primal/dual pair of each iteration runs as two
              hand-written Triton kernels (:mod:`triton_tvl1`), updating ``u``
              and ``p`` in place -- OpenCV's CUDA kernel split, but batched over
              ``(B, 2, H, W)``.  Exactly 2 launches per iteration against
              Inductor's ~6.9, and correspondingly less DRAM traffic.  The
              per-warp setup and the median network stay Inductor-compiled.
              Requires CUDA input tensors; ``gamma != 0`` is not implemented.
        early_exit: stop iterating a pyramid level as soon as the *whole
            batch* has converged, instead of always running the full
            ``warps * outer * inner`` trip count.  Adapted from OpenCV
            (tvl1flow.cpp:352-379): the batch-maximum error is read back to the
            host on throttled "check" iterations only -- after a check the next
            one is roughly ``error / (epsilon^2 h w)`` iterations away -- and
            the loop breaks when it falls to ``epsilon^2 h w``.  This is a pure
            speed win, **bit-identical** to ``early_exit=False``: the loop is
            only left once the per-sample freeze mask has frozen every sample,
            so the skipped iterations were no-ops.  Set to False to get static
            trip counts and no host syncs, which is what CUDA-graph capture
            needs; it is therefore forced to False for
            ``backend="cudagraphs"``.  Ignored when ``epsilon <= 0``.

    Returns:
        Flow tensor of shape ``(B, 2, H, W)``, same device/dtype as the
        input: channel 0 = x/u, channel 1 = y/v displacement in pixels, with
        ``prev[y, x] ~ next[y + v, x + u]`` (cv2 convention).
    """
    if prev.ndim != 3 or next.ndim != 3:
        raise ValueError(f"expected (B, H, W) inputs, got {tuple(prev.shape)} / {tuple(next.shape)}")
    if prev.shape != next.shape:
        raise ValueError("prev and next must have the same shape")
    if not (0.0 < scale_step < 1.0):
        raise ValueError("scale_step must be in (0, 1)")
    if nscales < 1:
        raise ValueError("nscales must be >= 1")
    if use_initial_flow and initial_flow is None:
        raise ValueError("use_initial_flow=True requires initial_flow")
    step_fn, warp_fn, median_fn = _resolve_backend(backend, prev.device)
    if backend == "cudagraphs":
        early_exit = False  # data-dependent trip counts defeat graph replay

    if not torch.is_floating_point(prev):
        prev = prev.to(torch.float32)
    imgs = torch.stack([prev, next.to(prev.dtype)], dim=1)  # (B, 2, H, W)
    B, _, H, W = imgs.shape
    dt, dev = imgs.dtype, imgs.device
    use_gamma = gamma != 0.0

    # Pyramid sizes: successive resize by scale_step (cvRound), clamped at 16.
    sizes: list[tuple[int, int]] = [(H, W)]
    for _s in range(1, nscales):
        h2 = int(round(sizes[-1][0] * scale_step))
        w2 = int(round(sizes[-1][1] * scale_step))
        if h2 < _MIN_SIZE or w2 < _MIN_SIZE:
            break
        sizes.append((h2, w2))
    ns = len(sizes)

    # Image pyramid: plain INTER_LINEAR chain, exactly like OpenCV (no
    # Gaussian pre-smoothing).  prev/next travel together as 2 channels.
    pyr = [imgs]
    for s in range(1, ns):
        pyr.append(F.interpolate(pyr[-1], size=sizes[s], mode="bilinear", align_corners=False))

    # Initial flow at the coarsest scale.
    if use_initial_flow:
        u = initial_flow.to(device=dev, dtype=dt)
        if u is initial_flow:
            u = u.clone()  # backend="triton" updates u in place; never alias the caller's
        for s in range(1, ns):
            u = F.interpolate(u, size=sizes[s], mode="bilinear", align_corners=False) * scale_step
    else:
        u = torch.zeros(B, 2, *sizes[-1], dtype=dt, device=dev)
    u3 = torch.zeros(B, 1, *sizes[-1], dtype=dt, device=dev) if use_gamma else None

    # Coarse-to-fine.
    for s in range(ns - 1, -1, -1):
        u, u3 = _proc_one_scale(
            pyr[s][:, 0:1],
            pyr[s][:, 1:2],
            u,
            u3,
            tau=tau,
            lambda_=lambda_,
            theta=theta,
            warps=warps,
            epsilon=epsilon,
            inner_iterations=inner_iterations,
            outer_iterations=outer_iterations,
            gamma=gamma,
            median_filtering=median_filtering,
            early_exit=early_exit,
            step_fn=step_fn,
            warp_fn=warp_fn,
            median_fn=median_fn,
        )
        if s > 0:
            u = F.interpolate(u, size=sizes[s - 1], mode="bilinear", align_corners=False) * (1.0 / scale_step)
            if use_gamma:
                u3 = F.interpolate(u3, size=sizes[s - 1], mode="bilinear", align_corners=False)

    return u


def calc_flow_tvl1_video(
    frames: torch.Tensor,
    *,
    chunk: int | None = None,
    **params,
) -> torch.Tensor:
    """Flow for every consecutive frame pair of one or many sequences.

    Exploits the batched pair API: all T-1 (or N*(T-1)) pairs are stacked
    into a single batch and solved in one ``calc_flow_tvl1`` call.

    Args:
        frames: ``(T, H, W)`` for a single sequence -- returns
            ``(T-1, 2, H, W)`` -- or ``(N, T, H, W)`` for N sequences of equal
            length -- returns ``(N, T-1, 2, H, W)``.  ``flow[..., t, :, :, :]``
            is the flow from frame t to frame t+1.  For the 3-D case the
            prev/next batches are views of ``frames`` (no copies).
        chunk: optional micro-batch size bounding peak memory: the flattened
            pair batch is processed ``chunk`` pairs at a time and the results
            concatenated.  ``None`` (default) = one single batch.
        **params: forwarded to :func:`calc_flow_tvl1` (including
            ``backend="eager" | "compile" | "cudagraphs"``).

    Returns:
        Flow tensor as described above (cv2 sign convention, pixels).
    """
    if frames.ndim == 3:
        prv, nxt = frames[:-1], frames[1:]  # views, no copy
    elif frames.ndim == 4:
        n, t, h, w = frames.shape
        prv = frames[:, :-1].reshape(n * (t - 1), h, w)
        nxt = frames[:, 1:].reshape(n * (t - 1), h, w)
    else:
        raise ValueError(f"expected (T, H, W) or (N, T, H, W) frames, got {tuple(frames.shape)}")
    if prv.shape[0] == 0:
        raise ValueError("need at least 2 frames")

    b = prv.shape[0]
    if chunk is None or chunk >= b:
        flow = calc_flow_tvl1(prv, nxt, **params)
    else:
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        flow = torch.cat(
            [calc_flow_tvl1(prv[i : i + chunk], nxt[i : i + chunk], **params) for i in range(0, b, chunk)],
            dim=0,
        )
    if frames.ndim == 4:
        return flow.view(frames.shape[0], frames.shape[1] - 1, 2, *frames.shape[2:])
    return flow


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    import numpy as np
    import cv2

    from benchmark_data import load_flow_gt, load_pairs

    torch.manual_seed(0)
    rng = np.random.default_rng(12345)

    def epe(f: np.ndarray, g: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Mean endpoint error between two (2, H, W) flow fields."""
        e = np.sqrt(((f - g) ** 2).sum(axis=0))
        return float(e[mask].mean() if mask is not None else e.mean())

    def cv_tvl1(p8: np.ndarray, n8: np.ndarray) -> np.ndarray:
        alg = cv2.optflow.DualTVL1OpticalFlow_create()  # defaults == ours
        return alg.calc(p8, n8, None).transpose(2, 0, 1)  # (2, H, W)

    ok = True

    # ---------------- median selection network == torch.median ----------------
    for k in (3, 5):
        xr = torch.randn(3, 2, 37, 41)
        xq = (torch.randint(0, 5, (2, 2, 19, 23)).float())  # many ties
        net_ok = all(
            torch.equal(_median_blur(t, k), _median_blur_sort(t, k)) for t in (xr, xq)
        )
        prog, nreg, _ = _NETWORKS[k * k]
        print(f"[median {k}x{k} network] {len(prog)} min/max ops, {nreg} registers, "
              f"exact match vs torch.median: {net_ok}")
        ok &= net_ok
    # cv2 parity (cv2.medianBlur supports 3/5 only on CV_32F); informational --
    # the gate above is the exact one, this only cross-checks border handling.
    xcv = rng.standard_normal((41, 53)).astype(np.float32)
    for k in (3, 5):
        ref = cv2.medianBlur(xcv, k)
        mine = _median_blur(torch.from_numpy(xcv)[None, None], k)[0, 0].numpy()
        r = k // 2
        print(f"[median {k}x{k} vs cv2.medianBlur] maxdiff = {np.abs(ref - mine).max():.1e} "
              f"(interior {np.abs(ref[r:-r, r:-r] - mine[r:-r, r:-r]).max():.1e})")

    # ---------------- synthetic known-shift pair ----------------
    Hs, Ws = 192, 256
    tex = cv2.GaussianBlur(rng.standard_normal((Hs, Ws)).astype(np.float32), (0, 0), 3.0)
    tex = tex + 2.0 * cv2.GaussianBlur(rng.standard_normal((Hs, Ws)).astype(np.float32), (0, 0), 8.0)
    tex = cv2.normalize(tex, None, 0.0, 255.0, cv2.NORM_MINMAX)
    prev_s = np.round(tex).astype(np.uint8)
    dx, dy = 2.3, -1.6
    ys_g, xs_g = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    next_s = cv2.remap(prev_s.astype(np.float32), xs_g - dx, ys_g - dy,
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    next_s = np.round(next_s).astype(np.uint8)
    gt_s = np.stack([np.full((Hs, Ws), dx, np.float32), np.full((Hs, Ws), dy, np.float32)])

    cv_s = cv_tvl1(prev_s, next_s)
    t0 = time.time()
    my_s = calc_flow_tvl1(
        torch.from_numpy(prev_s.astype(np.float32))[None],
        torch.from_numpy(next_s.astype(np.float32))[None],
    )[0].numpy()
    t_mine = time.time() - t0
    e_cv = epe(my_s, cv_s)
    print(f"[synthetic shift ({dx},{dy}) {Hs}x{Ws}]  ({t_mine:.1f}s torch-cpu)")
    print(f"  EPE mine-vs-cv2 = {e_cv:.4f} px | mine-vs-GT = {epe(my_s, gt_s):.4f} px"
          f" | cv2-vs-GT = {epe(cv_s, gt_s):.4f} px")
    ok &= e_cv < 0.5

    # ---------------- Middlebury pair (downscaled for CPU speed) ----------------
    mb_h, mb_w = 240, 320
    chosen = None
    for f1, f2, name in load_pairs(gray=True, size=(mb_h, mb_w)):
        gt = load_flow_gt(name)
        if gt is not None:
            chosen = (f1, f2, name, gt)
            break
    if chosen is None:
        print("[middlebury] no pair with ground truth found -- skipped")
    else:
        f1, f2, name, gt = chosen
        gt_full = gt.transpose(2, 0, 1)  # (2, H0, W0)
        fx_s, fy_s = mb_w / gt.shape[1], mb_h / gt.shape[0]
        gt_small = cv2.resize(gt, (mb_w, mb_h), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        gt_small[0] *= fx_s
        gt_small[1] *= fy_s
        valid = np.abs(gt_small).max(axis=0) < 1e9

        cv_m = cv_tvl1(f1, f2)
        t0 = time.time()
        my_m = calc_flow_tvl1(
            torch.from_numpy(f1.astype(np.float32))[None],
            torch.from_numpy(f2.astype(np.float32))[None],
        )[0].numpy()
        t_mine = time.time() - t0
        e_cv = epe(my_m, cv_m)
        print(f"[middlebury '{name}' {mb_h}x{mb_w}]  ({t_mine:.1f}s torch-cpu)")
        print(f"  EPE mine-vs-cv2 = {e_cv:.4f} px | mine-vs-GT = {epe(my_m, gt_small, valid):.4f} px"
              f" | cv2-vs-GT = {epe(cv_m, gt_small, valid):.4f} px")
        ok &= e_cv < 0.5

    # ---------------- batch consistency: B=4 copies == B=1 ----------------
    p4 = torch.from_numpy(np.stack([prev_s.astype(np.float32)] * 4))
    n4 = torch.from_numpy(np.stack([next_s.astype(np.float32)] * 4))
    f4 = calc_flow_tvl1(p4, n4)
    f1_t = calc_flow_tvl1(p4[:1], n4[:1])
    bdiff = float((f4 - f1_t).abs().max())
    print(f"[batch] max |B=4 - B=1| = {bdiff:.3e}")
    ok &= bdiff < 1e-4

    # ---------------- video wrapper: (T,H,W) and (N,T,H,W) ----------------
    # Cheap parameters: these tests check wrapper/batching self-consistency,
    # not cv2 parity (covered above).
    fast = dict(nscales=3, warps=2, inner_iterations=10, outer_iterations=3)
    Hv, Wv = 96, 128
    base = cv2.normalize(
        cv2.GaussianBlur(rng.standard_normal((Hv + 40, Wv + 40)).astype(np.float32), (0, 0), 3.0),
        None, 0.0, 255.0, cv2.NORM_MINMAX,
    )

    def seq(dx_step: float, dy_step: float, t: int) -> np.ndarray:
        return np.stack(
            [base[20 : 20 + Hv, 20 : 20 + Wv]] + [
                cv2.remap(
                    base,
                    (np.mgrid[0:Hv, 0:Wv][1] + 20 - k * dx_step).astype(np.float32),
                    (np.mgrid[0:Hv, 0:Wv][0] + 20 - k * dy_step).astype(np.float32),
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT,
                )
                for k in range(1, t)
            ]
        )

    frames_t = torch.from_numpy(seq(1.2, -0.8, 5))  # (T=5, H, W)
    fv = calc_flow_tvl1_video(frames_t, **fast)
    floop = torch.cat(
        [calc_flow_tvl1(frames_t[i : i + 1], frames_t[i + 1 : i + 2], **fast) for i in range(4)]
    )
    d_loop = float((fv - floop).abs().max())
    print(f"[video T=5] shape {tuple(fv.shape)}, max |video - per-pair loop| = {d_loop:.3e}")
    ok &= fv.shape == (4, 2, Hv, Wv) and d_loop < 1e-4

    frames_nt = torch.from_numpy(np.stack([seq(1.2, -0.8, 4), seq(-0.9, 1.1, 4)]))  # (N=2, T=4, H, W)
    fnt = calc_flow_tvl1_video(frames_nt, **fast)
    fnt_c = calc_flow_tvl1_video(frames_nt, chunk=2, **fast)
    floop2 = torch.stack(
        [
            torch.cat(
                [calc_flow_tvl1(frames_nt[n_, i : i + 1], frames_nt[n_, i + 1 : i + 2], **fast) for i in range(3)]
            )
            for n_ in range(2)
        ]
    )
    d_loop2 = float((fnt - floop2).abs().max())
    d_chunk = float((fnt - fnt_c).abs().max())
    print(
        f"[video N=2,T=4] shape {tuple(fnt.shape)}, max |video - loop| = {d_loop2:.3e}, "
        f"max |chunk=2 - chunk=None| = {d_chunk:.3e}"
    )
    ok &= fnt.shape == (2, 3, 2, Hv, Wv) and d_loop2 < 1e-4 and d_chunk < 1e-4

    # ---------------- backend="compile" parity (CPU) ----------------
    fp, fn_ = frames_t[0:1], frames_t[1:2]
    fe = calc_flow_tvl1(fp, fn_, **fast)
    t0 = time.time()
    fc = calc_flow_tvl1(fp, fn_, backend="compile", **fast)
    t_compile = time.time() - t0
    t0 = time.time()
    fc2 = calc_flow_tvl1(fp, fn_, backend="compile", **fast)  # warm (cached)
    t_warm = time.time() - t0
    cdiff = float((fe - fc).abs().max())
    cdiff2 = float((fc - fc2).abs().max())
    print(
        f"[compile] maxdiff vs eager = {cdiff:.3e} "
        f"(first call {t_compile:.1f}s incl. compile, warm {t_warm:.1f}s, rerun diff {cdiff2:.1e})"
    )
    ok &= torch.allclose(fe, fc, atol=1e-3, rtol=1e-4) and cdiff2 == 0.0

    # "cudagraphs" needs CUDA; on CPU the guard must reject it cleanly.
    if torch.cuda.is_available():
        fg = calc_flow_tvl1(fp.cuda(), fn_.cuda(), backend="cudagraphs", **fast).cpu()
        gdiff = float((fe - fg).abs().max())
        print(f"[cudagraphs] maxdiff vs eager = {gdiff:.3e}")
        ok &= torch.allclose(fe, fg, atol=1e-3, rtol=1e-4)
    else:
        try:
            calc_flow_tvl1(fp, fn_, backend="cudagraphs", **fast)
            print("[cudagraphs] ERROR: CPU call did not raise")
            ok = False
        except RuntimeError as e:
            print(f"[cudagraphs] no CUDA here; guard raised as expected ({e})")

    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
