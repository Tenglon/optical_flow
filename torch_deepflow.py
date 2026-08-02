"""Batched PyTorch reimplementation of OpenCV's DeepFlow optical flow.

This replicates ``cv2.optflow.createOptFlow_DeepFlow()`` (opencv_contrib,
``modules/optflow/src/deepflow.cpp``), which is only the *variational* part of
Weinzaepfel et al., "DeepFlow: Large displacement optical flow with deep
matching", ICCV 2013 (no DeepMatching initialization).  Internally OpenCV's
DeepFlow is a coarse-to-fine pyramid wrapper around
``cv::VariationalRefinement`` (opencv, ``modules/video/src/variational_refinement.cpp``)
called with remapped parameters ``alpha' = 4*alpha``, ``delta' = delta/3``,
``gamma' = gamma/3``; that solver is reproduced here operator-for-operator.

Minimized energy (per pyramid level, Brox-style, with w = (u, v)):

    E(w) = integral of
        delta' * Psi( |I1(x+w) - I0(x)|^2 / (|grad I|^2 + zeta^2) )      (color constancy, normalized)
      + gamma' * Psi( |grad I1(x+w) - grad I0(x)|^2 / (...) + zeta^2) )  (gradient constancy, normalized)
      + alpha' * Psi( |grad u|^2 + |grad v|^2 )                          (TV-like smoothness)

with the robust penalty Psi(s^2) = sqrt(s^2 + epsilon^2), epsilon = 0.001, and
data-term normalization constant zeta = 0.1 (both hard-coded in OpenCV).  Each
level runs `fixed_point_iterations` outer linearizations; every linearized
system is solved with red-black successive over-relaxation (`sor_iterations`
sweeps, relaxation factor `omega`), vectorized here via checkerboard masks.

All operations are pure torch, fully batched, and run unmodified on CUDA.
Images are expected in the 0..255 intensity range (OpenCV converts uint8 to
float without rescaling; the zeta normalization is not scale invariant).

Sign convention matches cv2: ``next(x + flow_u, y + flow_v) ~= prev(x, y)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["calc_flow_deepflow", "calc_flow_deepflow_video"]

# Constants hard-coded inside cv::VariationalRefinement
_ZETA = 0.1  # data-term normalization constant
_EPSILON = 0.001  # robust penalty regularizer of Psi(s^2) = sqrt(s^2 + eps^2)


# ---------------------------------------------------------------------------
# Small shift / derivative helpers (all zero-copy-free, batched over leading dims)
# ---------------------------------------------------------------------------

def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """y[..., j] = x[..., j-1]; first column zero."""
    return F.pad(x[..., :, :-1], (1, 0))


def _shift_left(x: torch.Tensor) -> torch.Tensor:
    """y[..., j] = x[..., j+1]; last column zero."""
    return F.pad(x[..., :, 1:], (0, 1))


def _shift_down(x: torch.Tensor) -> torch.Tensor:
    """y[..., i, :] = x[..., i-1, :]; first row zero."""
    return F.pad(x[..., :-1, :], (0, 0, 1, 0))


def _shift_up(x: torch.Tensor) -> torch.Tensor:
    """y[..., i, :] = x[..., i+1, :]; last row zero."""
    return F.pad(x[..., 1:, :], (0, 0, 0, 1))


def _fwd_dx(x: torch.Tensor) -> torch.Tensor:
    """Forward difference x[j+1]-x[j], zero in the last column (replicate border)."""
    return F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1))


def _fwd_dy(x: torch.Tensor) -> torch.Tensor:
    """Forward difference x[i+1]-x[i], zero in the last row (replicate border)."""
    return F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))


def _grad_h(x: torch.Tensor) -> torch.Tensor:
    """Central difference [-1, 0, 1] along x with replicate borders.

    Matches OpenCV ``Sobel(src, dst, -1, 1, 0, ksize=1, BORDER_REPLICATE)``
    (note: no 1/2 factor, exactly as in variational_refinement.cpp).
    """
    xp = F.pad(x.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(1)
    return xp[..., :, 2:] - xp[..., :, :-2]


def _grad_v(x: torch.Tensor) -> torch.Tensor:
    """Central difference [-1, 0, 1] along y with replicate borders (Sobel ksize=1)."""
    xp = F.pad(x.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
    return xp[..., 2:, :] - xp[..., :-2, :]


def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian pre-smoothing, replicating OpenCV DeepFlow.

    Kernel length is ``2*floor(3*sigma) + 1`` and coefficients follow
    ``cv::getGaussianKernel``; the border mode is BORDER_REFLECT_101
    (= torch 'reflect').  ``img`` is (B, H, W).
    """
    klen = 2 * int(math.floor(3.0 * sigma)) + 1
    if klen <= 1 or sigma <= 0:
        return img
    i = torch.arange(klen, dtype=torch.float32, device=img.device) - (klen - 1) / 2.0
    k = torch.exp(-(i * i) / (2.0 * sigma * sigma))
    k = (k / k.sum()).to(img.dtype)
    r = klen // 2
    x = img.unsqueeze(1)  # (B, 1, H, W)
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, klen))
    x = F.conv2d(x, k.view(1, 1, klen, 1))
    return x.squeeze(1)


def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize matching cv::resize INTER_LINEAR (half-pixel centers)."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _warp(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``img`` by flow: out(x, y) = img(x + u, y + v).

    Bilinear with clamped (replicate) borders, matching OpenCV
    ``remap(..., INTER_LINEAR, BORDER_REPLICATE)``.  ``img, u, v`` are (B, H, W).
    """
    B, H, W = img.shape
    dev, dt = img.device, img.dtype
    gy, gx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=dt),
        torch.arange(W, device=dev, dtype=dt),
        indexing="ij",
    )
    x = gx + u
    y = gy + v
    xn = 2.0 * x / max(W - 1, 1) - 1.0
    yn = 2.0 * y / max(H - 1, 1) - 1.0
    grid = torch.stack((xn, yn), dim=-1)  # (B, H, W, 2)
    out = F.grid_sample(
        img.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(1)


# ---------------------------------------------------------------------------
# Hot inner steps (pure tensor functions, torch.compile-able)
# ---------------------------------------------------------------------------

def _build_system(
    du, dv, cu, cv_,
    Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz,
    dnorm, dnorm_x, dnorm_y,
    dx_u0, dy_u0, dx_v0, dy_v0,
    hmask, vmask,
    delta2: float, gamma2: float, alpha2: float,
):
    """Rebuild the linearized system for one fixed-point iteration.

    Pure tensor function (no data-dependent Python branching) so it can be
    wrapped with ``torch.compile``.  Returns the system coefficients
    (A11, A12, A22, b1, b2) and the SOR neighbor weights (wL, wC, wT).
    """
    zeta2 = _ZETA * _ZETA
    eps2 = _EPSILON * _EPSILON

    # ---- data term: robust color + gradient constancy, linearized at du, dv ----
    # color constancy:  Psi'( (Iz + Ix du + Iy dv)^2 / dnorm ) / dnorm
    ik1z = Iz + Ix * du + Iy * dv
    wd = delta2 / (torch.sqrt(ik1z * ik1z / dnorm + eps2) * dnorm)
    A11 = wd * (Ix * Ix) + zeta2
    A12 = wd * (Ix * Iy)
    A22 = wd * (Iy * Iy) + zeta2
    b1 = -wd * (Iz * Ix)
    b2 = -wd * (Iz * Iy)
    # gradient constancy
    ik1zx = Ixz + Ixx * du + Ixy * dv
    ik1zy = Iyz + Ixy * du + Iyy * dv
    wg = gamma2 / torch.sqrt(ik1zx * ik1zx / dnorm_x + ik1zy * ik1zy / dnorm_y + eps2)
    A11 = A11 + wg * (Ixx * Ixx / dnorm_x + Ixy * Ixy / dnorm_y)
    A12 = A12 + wg * (Ixx * Ixy / dnorm_x + Ixy * Iyy / dnorm_y)
    A22 = A22 + wg * (Ixy * Ixy / dnorm_x + Iyy * Iyy / dnorm_y)
    b1 = b1 - wg * (Ixx * Ixz / dnorm_x + Ixy * Iyz / dnorm_y)
    b2 = b2 - wg * (Ixy * Ixz / dnorm_x + Iyy * Iyz / dnorm_y)

    # ---- smoothness term: TV weight from the current flow estimate ----
    # w(i,j) = (alpha/2) / |grad w|, shared by the two forward edges
    # (i,j)-(i,j+1) and (i,j)-(i+1,j), forward differences, replicate border.
    ux, uy = _fwd_dx(cu), _fwd_dy(cu)
    vx, vy = _fwd_dx(cv_), _fwd_dy(cv_)
    wS = alpha2 / torch.sqrt(ux * ux + vx * vx + uy * uy + vy * vy + eps2)
    wh = wS * hmask  # horizontal edge weights (no edge out of last column)
    wv = wS * vmask  # vertical edge weights (no edge out of last row)

    # add smoothness contributions of the *level-initial* flow (u0, v0) to b,
    # and the edge weights to the diagonal A11/A22 (A12 gets none)
    fx_u, fy_u = wh * dx_u0, wv * dy_u0
    fx_v, fy_v = wh * dx_v0, wv * dy_v0
    wsum = wh + _shift_right(wh) + wv + _shift_down(wv)
    A11 = A11 + wsum
    A22 = A22 + wsum
    b1 = b1 + fx_u - _shift_right(fx_u) + fy_u - _shift_down(fy_u)
    b2 = b2 + fx_v - _shift_right(fx_v) + fy_v - _shift_down(fy_v)

    # SOR neighbor weights: left edge w(i,j-1), top edge w(i-1,j),
    # right/bottom edge w(i,j); out-of-domain terms vanish.
    wL = _shift_right(wS).unsqueeze(1)  # (B,1,H,W)
    wT = _shift_down(wS).unsqueeze(1)
    wC = wS.unsqueeze(1)
    return A11, A12, A22, b1, b2, wL, wC, wT


def _sor_step(d, wL, wC, wT, A11, A12, A22, b1, b2, red, black, omega: float):
    """One full red-black SOR sweep (red update, then black) for d = (du, dv).

    This is the hot inner step, executed sor_iterations times per fixed-point
    iteration.  Pure tensor function with a statically unrolled 2-color loop,
    no data-dependent branching, so it is safe to torch.compile.
    """
    for mask in (red, black):
        sig = (
            wL * _shift_right(d)
            + wC * _shift_left(d)
            + wT * _shift_down(d)
            + wC * _shift_up(d)
        )
        sig_u, sig_v = sig[:, 0], sig[:, 1]
        du, dv = d[:, 0], d[:, 1]
        du_new = du + omega * ((sig_u + b1 - dv * A12) / A11 - du)
        du = torch.where(mask, du_new, du)
        # dv update uses the just-updated du (as in OpenCV's inner loop)
        dv_new = dv + omega * ((sig_v + b2 - du * A12) / A22 - dv)
        dv = torch.where(mask, dv_new, dv)
        d = torch.stack((du, dv), dim=1)
    return d


# Lazily created torch.compile wrappers of the hot steps, keyed by backend.
# We deliberately compile with dynamic=False: the SOR step is launch-bound and
# the ~44 pyramid shapes recur on every call, so each (shape, B) specializes
# once into its own fast graph and is served from the dynamo cache afterwards
# (cache limits are raised below to hold all per-level variants).  We chose
# per-shape specialization over marking dims dynamic or shape bucketing:
# recompiles are a one-time warm-up cost, while static shapes give Inductor /
# CUDA graphs the most freedom (and "cudagraphs" requires static shapes anyway).
_COMPILED_FNS: dict[str, tuple] = {}


def _resolve_backend(backend: str):
    """Return (build_system_fn, sor_step_fn) for the requested backend."""
    if backend == "eager":
        return _build_system, _sor_step
    if backend in _COMPILED_FNS:
        return _COMPILED_FNS[backend]
    if backend not in ("compile", "cudagraphs"):
        raise ValueError(
            f"backend must be 'eager', 'compile' or 'cudagraphs', got {backend!r}"
        )
    import torch._dynamo as _dynamo

    # one graph per (pyramid-level shape x batch size); make room for all of them
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 512)
    if hasattr(_dynamo.config, "accumulated_cache_size_limit"):
        _dynamo.config.accumulated_cache_size_limit = max(
            _dynamo.config.accumulated_cache_size_limit, 2048
        )
    if backend == "compile":
        fns = (
            torch.compile(_build_system, dynamic=False),
            torch.compile(_sor_step, dynamic=False),
        )
    else:  # "cudagraphs": CUDA-graph capture to eliminate launch overhead
        fns = tuple(
            _cudagraph_safe(torch.compile(f, mode="reduce-overhead", dynamic=False))
            for f in (_build_system, _sor_step)
        )
    _COMPILED_FNS[backend] = fns
    return fns


def _cudagraph_safe(fn):
    """Make a reduce-overhead-compiled fn safe for feed-back loops.

    CUDA-graph outputs live in static buffers that the next replay
    overwrites; we mark each call as a new step and clone the outputs so
    results that escape the iteration loop (e.g. a level's final du/dv)
    stay valid.
    """
    def wrapped(*args):
        torch.compiler.cudagraph_mark_step_begin()
        out = fn(*args)
        return tuple(o.clone() for o in out) if isinstance(out, tuple) else out.clone()
    return wrapped


# ---------------------------------------------------------------------------
# Single-level variational refinement (port of cv::VariationalRefinement)
# ---------------------------------------------------------------------------

def _variational_refinement(
    I0: torch.Tensor,
    I1: torch.Tensor,
    flow: torch.Tensor,
    *,
    alpha: float,
    delta: float,
    gamma: float,
    fixed_point_iterations: int,
    sor_iterations: int,
    omega: float,
    sys_fn=_build_system,
    sor_fn=_sor_step,
) -> torch.Tensor:
    """Refine ``flow`` (B, 2, H, W) on one pyramid level.

    Direct port of ``cv::VariationalRefinement::calcUV``: warp once per level,
    then ``fixed_point_iterations`` relinearizations of the robust data /
    smoothness weights, each solved by ``sor_iterations`` red-black SOR sweeps
    for the flow increment (du, dv).  ``sys_fn`` / ``sor_fn`` are the (possibly
    torch.compile-wrapped) hot inner steps.
    """
    B, H, W = I0.shape
    u0, v0 = flow[:, 0], flow[:, 1]

    zeta2 = _ZETA * _ZETA
    delta2 = delta / 2.0  # per-term weights as in OpenCV (delta/2, gamma/2, alpha/2)
    gamma2 = gamma / 2.0
    alpha2 = alpha / 2.0

    # --- image derivatives, computed once per level on the warped average ---
    warped = _warp(I1, u0, v0)
    Iz = warped - I0  # temporal derivative
    avg = 0.5 * (I0 + warped)
    Ix, Iy = _grad_h(avg), _grad_v(avg)
    Ixz, Iyz = _grad_h(Iz), _grad_v(Iz)
    Ixx, Ixy, Iyy = _grad_h(Ix), _grad_v(Ix), _grad_v(Iy)

    # Precomputed data-term normalization factors (Brox/DeepFlow "beta" trick)
    dnorm = Ix * Ix + Iy * Iy + zeta2
    dnorm_x = Ixx * Ixx + Ixy * Ixy + zeta2
    dnorm_y = Iyy * Iyy + Ixy * Ixy + zeta2

    # red = (i + j) % 2 == 0 checkerboard; red pixels only have black neighbors,
    # so a masked simultaneous update is exactly red-black Gauss-Seidel/SOR.
    ii = torch.arange(H, device=I0.device).view(H, 1)
    jj = torch.arange(W, device=I0.device).view(1, W)
    red = ((ii + jj) % 2 == 0)  # (H, W), broadcasts over batch
    black = ~red

    # masks zeroing the last column / row: smoothness edges only exist inside
    hmask = torch.ones(1, H, W, device=I0.device, dtype=I0.dtype)
    hmask[..., -1] = 0.0
    vmask = torch.ones(1, H, W, device=I0.device, dtype=I0.dtype)
    vmask[:, -1, :] = 0.0

    du = torch.zeros_like(u0)
    dv = torch.zeros_like(v0)
    cu, cv_ = u0, v0  # current flow estimate (tempW in OpenCV)

    dx_u0, dy_u0 = _fwd_dx(u0), _fwd_dy(u0)
    dx_v0, dy_v0 = _fwd_dx(v0), _fwd_dy(v0)

    for _ in range(fixed_point_iterations):
        # linearize data + smoothness terms around (du, dv) / current flow
        A11, A12, A22, b1, b2, wL, wC, wT = sys_fn(
            du, dv, cu, cv_,
            Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz,
            dnorm, dnorm_x, dnorm_y,
            dx_u0, dy_u0, dx_v0, dy_v0,
            hmask, vmask,
            delta2, gamma2, alpha2,
        )

        # ---- red-black SOR on the linearized system for (du, dv) ----
        d = torch.stack((du, dv), dim=1)  # (B,2,H,W)
        for _s in range(sor_iterations):
            d = sor_fn(d, wL, wC, wT, A11, A12, A22, b1, b2, red, black, omega)
        du, dv = d[:, 0], d[:, 1]

        cu = u0 + du
        cv_ = v0 + dv

    return torch.stack((cu, cv_), dim=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calc_flow_deepflow(
    prev: torch.Tensor,
    next: torch.Tensor,
    *,
    sigma: float = 0.6,
    min_size: int = 25,
    downscale_factor: float = 0.95,
    fixed_point_iterations: int = 5,
    sor_iterations: int = 25,
    alpha: float = 1.0,
    delta: float = 0.5,
    gamma: float = 5.0,
    omega: float = 1.6,
    backend: str = "eager",
) -> torch.Tensor:
    """Dense optical flow, replicating ``cv2.optflow.createOptFlow_DeepFlow``.

    Args:
        prev, next: (B, H, W) float grayscale images in the 0..255 range
            (matching OpenCV's uint8 -> CV_32F conversion without rescaling).
        sigma: Gaussian pre-smoothing std.
        min_size: minimum pyramid image dimension (levels stop before <= min_size).
        downscale_factor: pyramid scale per level.
        fixed_point_iterations: outer linearizations per level.
        sor_iterations: red-black SOR sweeps per linearization.
        alpha: smoothness weight; delta: color constancy weight; gamma:
            gradient constancy weight (DeepFlow parameterization; internally
            remapped to alpha*4, delta/3, gamma/3 as OpenCV does).
        omega: SOR relaxation factor.
        backend: execution backend for the hot inner steps (the per-iteration
            system rebuild and the red-black SOR sweep, ~44 levels x 5 x 25
            executions).  "eager" (default, byte-identical reference path),
            "compile" (torch.compile(dynamic=False); each pyramid-level shape
            specializes once and is then served from the dynamo cache), or
            "cudagraphs" (torch.compile mode="reduce-overhead", CUDA-only;
            eliminates kernel-launch overhead via CUDA graph replay).  Outer
            pyramid / fixed-point / SOR loops always stay eager, so tracing
            never unrolls them.

    Returns:
        (B, 2, H, W) flow in pixels, cv2 sign convention: channel 0 = u (x),
        channel 1 = v (y), with ``next(x + u, y + v) ~= prev(x, y)``.
    """
    if prev.dim() != 3 or next.dim() != 3 or prev.shape != next.shape:
        raise ValueError("prev and next must both be (B, H, W) with equal shapes")
    if backend == "cudagraphs" and not prev.is_cuda:
        raise RuntimeError("backend='cudagraphs' requires CUDA input tensors")
    sys_fn, sor_fn = _resolve_backend(backend)
    dev = prev.device
    I0 = prev.to(torch.float32)
    I1 = next.to(torch.float32)
    B, H, W = I0.shape

    # pre-smooth full-resolution images (once), then build the pyramid
    I0 = _gaussian_blur(I0, sigma)
    I1 = _gaussian_blur(I1, sigma)

    # pyramid sizes: next = round(prev * f); stop before any dim <= min_size
    sizes: list[tuple[int, int]] = [(H, W)]
    h, w = H, W
    while True:
        nh = int(h * downscale_factor + 0.5)
        nw = int(w * downscale_factor + 0.5)
        if nh <= min_size or nw <= min_size:
            break
        sizes.append((nh, nw))
        h, w = nh, nw

    # images resized successively from the previous level (as in OpenCV)
    pyr0 = [I0]
    pyr1 = [I1]
    for s in sizes[1:]:
        pyr0.append(_resize(pyr0[-1].unsqueeze(1), s).squeeze(1))
        pyr1.append(_resize(pyr1[-1].unsqueeze(1), s).squeeze(1))

    vr_kwargs = dict(
        alpha=4.0 * alpha,  # OpenCV DeepFlow -> VariationalRefinement remapping
        delta=delta / 3.0,
        gamma=gamma / 3.0,
        fixed_point_iterations=fixed_point_iterations,
        sor_iterations=sor_iterations,
        omega=omega,
        sys_fn=sys_fn,
        sor_fn=sor_fn,
    )

    flow = torch.zeros(B, 2, *sizes[-1], device=dev, dtype=torch.float32)
    for level in range(len(sizes) - 1, -1, -1):
        flow = _variational_refinement(pyr0[level], pyr1[level], flow, **vr_kwargs)
        if level > 0:
            flow = _resize(flow, sizes[level - 1]) * (1.0 / downscale_factor)
    return flow


def calc_flow_deepflow_video(
    frames: torch.Tensor,
    *,
    chunk: int | None = None,
    backend: str = "eager",
    **params,
) -> torch.Tensor:
    """DeepFlow between all consecutive frame pairs, as one batched call.

    Args:
        frames: either (T, H, W) — a single sequence — or (N, T, H, W) —
            N independent sequences of equal length.  Float grayscale, 0..255.
        chunk: optional micro-batch size bounding peak memory: the flattened
            pair batch is split into chunks of at most this size and the
            results concatenated.  None (default) = one single batch.
        backend: "eager" | "compile" | "cudagraphs", see :func:`calc_flow_deepflow`.
        **params: forwarded to :func:`calc_flow_deepflow` (sigma, alpha, ...).

    Returns:
        (T-1, 2, H, W) for (T, H, W) input, or (N, T-1, 2, H, W) for
        (N, T, H, W) input: ``out[..., t, :, :, :]`` is the flow from frame t
        to frame t+1 (cv2 sign convention).

    ``frames[:-1]`` / ``frames[1:]`` are taken as views (zero-copy); for the
    (N, T, H, W) case flattening the N and T-1 axes into one batch axis
    necessarily materializes the pair tensors once.
    """
    if frames.dim() not in (3, 4):
        raise ValueError("frames must be (T, H, W) or (N, T, H, W)")
    if frames.shape[-3] < 2:
        raise ValueError("need at least 2 frames per sequence")

    if frames.dim() == 3:
        prev, next_ = frames[:-1], frames[1:]  # views, zero-copy
        batched = False
    else:
        N, T, H, W = frames.shape
        # slicing along dim 1 keeps these as views; reshape to a flat batch
        # (N*(T-1), H, W) has to copy since the sliced strides are not mergeable
        prev = frames[:, :-1].reshape(-1, H, W)
        next_ = frames[:, 1:].reshape(-1, H, W)
        batched = True

    B = prev.shape[0]
    if chunk is None or chunk >= B:
        flow = calc_flow_deepflow(prev, next_, backend=backend, **params)
    else:
        if chunk < 1:
            raise ValueError("chunk must be a positive integer")
        flow = torch.cat(
            [
                calc_flow_deepflow(
                    prev[i : i + chunk], next_[i : i + chunk], backend=backend, **params
                )
                for i in range(0, B, chunk)
            ],
            dim=0,
        )

    if batched:
        flow = flow.view(N, T - 1, 2, H, W)
    return flow


# ---------------------------------------------------------------------------
# Self-test: compare against cv2.optflow DeepFlow and Middlebury ground truth
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    import cv2
    import numpy as np

    from benchmark_data import load_pairs, load_flow_gt

    torch.manual_seed(0)

    def epe(f_a: np.ndarray, f_b: np.ndarray, mask: np.ndarray | None = None) -> float:
        d = np.sqrt(((f_a - f_b) ** 2).sum(axis=-1))
        if mask is not None:
            d = d[mask]
        return float(d.mean())

    df = cv2.optflow.createOptFlow_DeepFlow()

    # ---- synthetic known-shift pair ----
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, size=(140, 180)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    dx_true, dy_true = 3, 2
    f1s = base.astype(np.uint8)
    # np.roll by (+dy, +dx): next(y, x) = prev(y - dy, x - dx), i.e.
    # next(y + dy, x + dx) = prev(y, x)  ->  GT flow (u, v) = (dx, dy)
    f2s = np.roll(f1s, (dy_true, dx_true), axis=(0, 1))
    flow_cv_s = df.calc(f1s, f2s, None)
    t1 = torch.from_numpy(f1s.astype(np.float32)).unsqueeze(0)
    t2 = torch.from_numpy(f2s.astype(np.float32)).unsqueeze(0)
    flow_t_s = calc_flow_deepflow(t1, t2)[0].permute(1, 2, 0).numpy()
    gt_s = np.zeros_like(flow_cv_s)
    gt_s[..., 0], gt_s[..., 1] = dx_true, dy_true
    inner = np.zeros(f1s.shape, bool)
    inner[10:-10, 10:-10] = True
    print("synthetic shift (+3,+2):")
    print(f"  EPE  torch vs cv2 : {epe(flow_t_s, flow_cv_s):.4f}")
    print(f"  EPE  cv2   vs GT  : {epe(flow_cv_s, gt_s, inner):.4f} (interior)")
    print(f"  EPE  torch vs GT  : {epe(flow_t_s, gt_s, inner):.4f} (interior)")

    # ---- Middlebury pairs (downscaled to keep runtime sane) ----
    HW = (240, 320)
    pairs = load_pairs(max_pairs=4, gray=True, size=HW)
    names = [n for _, _, n in pairs]
    print(f"\nMiddlebury pairs {names} at {HW[0]}x{HW[1]}:")

    cv_flows = []
    t0 = time.time()
    for f1, f2, _ in pairs:
        cv_flows.append(df.calc(f1, f2, None))
    t_cv = time.time() - t0

    batch1 = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float32))
    batch2 = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.float32))
    t0 = time.time()
    flows_t = calc_flow_deepflow(batch1, batch2)  # B = 4
    t_torch = time.time() - t0
    flows_t_np = flows_t.permute(0, 2, 3, 1).numpy()

    epes_cv = []
    for i, (f1, f2, name) in enumerate(pairs):
        e = epe(flows_t_np[i], cv_flows[i])
        epes_cv.append(e)
        line = f"  {name:<12} EPE torch vs cv2: {e:.4f}"
        gt = load_flow_gt(name)
        if gt is not None:
            H0, W0 = gt.shape[:2]
            valid = np.abs(gt).max(axis=-1) < 1e3
            gt0 = np.where(valid[..., None], gt, 0.0).astype(np.float32)
            gt_r = cv2.resize(gt0, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
            vfrac = cv2.resize(valid.astype(np.float32), (HW[1], HW[0]),
                               interpolation=cv2.INTER_AREA)
            gt_r /= np.maximum(vfrac, 1e-6)[..., None]  # normalized convolution
            gt_r[..., 0] *= HW[1] / W0
            gt_r[..., 1] *= HW[0] / H0
            valid_r = vfrac > 0.999
            line += (f" | EPE vs GT: cv2 {epe(cv_flows[i], gt_r, valid_r):.4f}"
                     f", torch {epe(flows_t_np[i], gt_r, valid_r):.4f}")
        print(line)
    print(f"  mean EPE torch vs cv2: {np.mean(epes_cv):.4f}"
          f"   (cv2 {t_cv:.1f}s for 4 pairs, torch B=4 {t_torch:.1f}s)")

    # ---- batch consistency: B=1 result must equal the B=4 slice ----
    single = calc_flow_deepflow(batch1[:1], batch2[:1])
    max_diff = (single[0] - flows_t[0]).abs().max().item()
    print(f"\nbatch consistency: max |B=1 - B=4| = {max_diff:.2e}")
    assert max_diff < 1e-4, "batch inconsistency"

    # ---- video wrapper: (T, H, W) and (N, T, H, W), plus chunking ----
    def make_seq(seed: int, t: int) -> torch.Tensor:
        r = np.random.default_rng(seed)
        b = r.uniform(0, 255, size=(90, 110)).astype(np.float32)
        b = cv2.GaussianBlur(b, (0, 0), 3.0)
        b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        seq = [np.roll(b, (2 * k, 3 * k), axis=(0, 1)) for k in range(t)]
        return torch.from_numpy(np.stack(seq).astype(np.float32))

    seq = make_seq(1, 5)  # (T=5, H, W)
    v = calc_flow_deepflow_video(seq)
    assert v.shape == (4, 2, 90, 110), v.shape
    loop = torch.cat(
        [calc_flow_deepflow(seq[t : t + 1], seq[t + 1 : t + 2]) for t in range(4)]
    )
    d_seq = (v - loop).abs().max().item()
    assert torch.allclose(v, loop), "video (T,H,W) != pairwise loop"

    seqs = torch.stack([make_seq(2, 4), make_seq(3, 4)])  # (N=2, T=4, H, W)
    vn = calc_flow_deepflow_video(seqs)
    assert vn.shape == (2, 3, 2, 90, 110), vn.shape
    loop_n = torch.stack(
        [
            torch.cat(
                [
                    calc_flow_deepflow(seqs[n, t : t + 1], seqs[n, t + 1 : t + 2])
                    for t in range(3)
                ]
            )
            for n in range(2)
        ]
    )
    d_nt = (vn - loop_n).abs().max().item()
    assert torch.allclose(vn, loop_n), "video (N,T,H,W) != pairwise loop"

    vc = calc_flow_deepflow_video(seqs, chunk=2)
    d_chunk = (vc - vn).abs().max().item()
    assert torch.allclose(vc, vn), "chunk=2 != chunk=None"
    print(
        "video wrapper: max |video - loop| (T,H,W) = "
        f"{d_seq:.2e}, (N,T,H,W) = {d_nt:.2e}, |chunk=2 - chunk=None| = {d_chunk:.2e}"
    )

    # ---- backend="compile": CPU parity with the eager reference path ----
    # small pair -> ~18 pyramid shapes, so per-shape dynamic=False compiles
    # stay cheap on the login node; "cudagraphs" is CUDA-only (guarded).
    # Inductor CPU codegen needs g++ >= 10 (-std=c++20); the system default on
    # this cluster is gcc 8.5, so look for a newer one (e.g. OpenHPC gnu12).
    def _pick_modern_cxx() -> str | None:
        import glob
        import os
        import shutil
        import subprocess

        cands = [os.environ.get("CXX"), shutil.which("g++")]
        cands += sorted(glob.glob("/opt/ohpc/pub/compiler/gcc/*/bin/g++"), reverse=True)
        cands += sorted(glob.glob("/opt/rh/gcc-toolset-*/root/usr/bin/g++"), reverse=True)
        for c in cands:
            if not c or not os.path.isfile(c):
                continue
            try:
                ver = subprocess.run([c, "-dumpversion"], capture_output=True,
                                     text=True, timeout=10).stdout.strip()
                if int(ver.split(".")[0]) >= 10:
                    return c
            except Exception:
                continue
        return None

    cxx = _pick_modern_cxx()
    if cxx is None:
        print("\nbackend='compile': SKIPPED (no g++ >= 10 found for Inductor CPU)")
    else:
        import os

        os.environ["CXX"] = cxx
        os.environ["PATH"] = os.path.dirname(cxx) + os.pathsep + os.environ["PATH"]
    rc = np.random.default_rng(7)
    small = rc.uniform(0, 255, size=(60, 80)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX)
    s1 = torch.from_numpy(small)
    s2 = torch.from_numpy(np.roll(small, (1, 2), axis=(0, 1)).copy())
    fe = calc_flow_deepflow(s1[None], s2[None])  # eager reference
    if cxx is not None:
        t0 = time.time()
        fc = calc_flow_deepflow(s1[None], s2[None], backend="compile")
        t_warm = time.time() - t0
        t0 = time.time()
        fc2 = calc_flow_deepflow(s1[None], s2[None], backend="compile")
        t_hot = time.time() - t0
        d_compile = (fc - fe).abs().max().item()
        print(f"\nbackend='compile' (CPU): max |compile - eager| = {d_compile:.2e}"
              f" (warm-up {t_warm:.1f}s incl. per-shape compiles, cached {t_hot:.2f}s)")
        assert d_compile < 1e-3, "compile backend diverges from eager"
        assert torch.equal(fc, fc2), "compiled backend not deterministic"
    try:
        calc_flow_deepflow(s1[None], s2[None], backend="cudagraphs")
        raise AssertionError("cudagraphs on CPU should have been rejected")
    except RuntimeError as e:
        print(f"backend='cudagraphs' on CPU correctly rejected: {e}")

    ok = np.mean(epes_cv) < 0.5
    print("PASS" if ok else "FAIL: mean EPE vs cv2 >= 0.5")
