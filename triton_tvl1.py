"""Hand-fused Triton kernels for the batched TV-L1 primal-dual iteration.

Mirrors the kernel split of OpenCV's CUDA TV-L1
(``opencv_contrib/modules/cudaoptflow/src/cuda/tvl1flow.cu``) -- two kernels
per iteration, both updating their state **in place** --

* ``_estimate_u_kernel``  (their ``estimateUKernel``): thresholding (v-step)
  + backward-difference divergence of ``p`` + primal update ``u = v + theta*div p``
  + (optionally) the per-pixel squared update for the convergence reduction
  + the per-sample freeze-mask select;
* ``_estimate_dual_kernel``  (their ``estimateDualVariablesKernel``): forward
  differences of ``u`` + dual ascent + reprojection of the four ``p`` planes.

...but batched: tensors are ``(B, 2, H, W)`` and the launch grid is
``(ceil(W/BW), ceil(H/BH), B)``, i.e. one program per (batch sample, tile), so
the coarse pyramid levels that leave an L40 74 % idle for OpenCV (see
``analysis_opencv_cuda.md`` §2.3) are filled by the batch axis instead.

Border semantics are bit-compatible with ``torch_flow.py``:

* ``_divergence``: ``div(y,x) = p1(y,x) - p1(y,x-1) + p2(y,x) - p2(y-1,x)`` with
  the out-of-range terms taken as **0** (not OpenCV's edge special-casing);
* ``_forward_gradient``: ``dx(y,W-1) = 0``, ``dy(H-1,x) = 0``.

Convergence reduction.  ``torch_flow`` needs, per batch sample, the sum of the
squared primal update over ``(C,H,W)``, then the batch maximum, then the freeze
mask.  Doing that with atomics would make the value depend on the (arbitrary)
order in which tiles retire, so the freeze *timing* -- and hence the result --
would stop being reproducible across batch sizes.  Instead
``_estimate_u_kernel`` writes one deterministic partial per tile into an
``(B, n_tiles)`` scratch plane and ``_estimate_dual_kernel``'s ``(tile 0, b)``
program folds those partials (a fixed-order loop over fixed-size ``tl.sum``
chunks), writes ``err[b]`` and the *next* freeze mask.  Because the mask is
double-buffered, the other programs of the same launch keep reading the current
one, so this costs **no extra launch**: still exactly 2 kernels per iteration,
matching OpenCV, with the batch-max left to the host (it is only needed on the
throttled ``.item()`` sync iterations).

Compute is always done in fp32 (loads are promoted, stores are cast back), so
the fp16/bf16 variants keep an fp32 solve with half-precision *traffic* -- which
is what a bandwidth-bound stencil wants, and strictly more accurate than
``torch_flow``'s reduced-precision path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

__all__ = ["TritonTVL1Step"]

# (BH, BW, num_warps) -- modest set, timed once per (H, W, dtype); see
# `_pick_config`.  BW >= 32 keeps the innermost (W) axis coalesced.
_CONFIGS: tuple[tuple[int, int, int], ...] = (
    (8, 32, 2),
    (8, 64, 4),
    (16, 64, 8),
)


# --------------------------------------------------------------------------- #
# Kernel 1: thresholding + divergence + primal update (+ error) -- in place
# --------------------------------------------------------------------------- #
@triton.jit
def _estimate_u_kernel(
    u_ptr,            # (B, 2, H, W) in/out
    px_ptr, py_ptr,   # (B, 2, H, W) dual field: px = (p11, p21), py = (p12, p22)
    i1w_ptr,          # (B, 2, H, W) warped gradient (I1wx, I1wy)
    rho_c_ptr, thr_ptr, pos_ptr, sg_ptr,   # (B, 1, H, W)
    active_ptr,       # (B,) bool -- per-sample freeze mask
    errp_ptr,         # (B, NT) fp32 per-tile partial sums (CALC_ERR only)
    H, W, HW, CHW,    # CHW = 2*H*W
    NW, NT,           # tiles along W, tiles per sample
    l_t, theta,
    BH: tl.constexpr, BW: tl.constexpr, CALC_ERR: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = tl.program_id(2)

    offh = pid_h * BH + tl.arange(0, BH)
    offw = pid_w * BW + tl.arange(0, BW)
    hin = offh[:, None] < H
    win = offw[None, :] < W
    m = hin & win
    idx = offh[:, None] * W + offw[None, :]
    b2 = b * CHW
    b1 = b * HW

    act = tl.load(active_ptr + b).to(tl.int1)

    u1 = tl.load(u_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    u2 = tl.load(u_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    ix = tl.load(i1w_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    iy = tl.load(i1w_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    rc = tl.load(rho_c_ptr + b1 + idx, mask=m, other=0.0).to(tl.float32)
    th = tl.load(thr_ptr + b1 + idx, mask=m, other=0.0).to(tl.float32)
    ps = tl.load(pos_ptr + b1 + idx, mask=m, other=0).to(tl.int1)
    sg = tl.load(sg_ptr + b1 + idx, mask=m, other=1.0).to(tl.float32)

    # ---- thresholding step: v = u + shrink(rho) * grad I1w
    rho = rc + ix * u1 + iy * u2
    fi = -rho / sg
    lo = rho < -th
    hi = rho > th
    d1 = tl.where(lo, l_t * ix, tl.where(hi, -l_t * ix, tl.where(ps, fi * ix, 0.0)))
    d2 = tl.where(lo, l_t * iy, tl.where(hi, -l_t * iy, tl.where(ps, fi * iy, 0.0)))

    # ---- divergence (backward differences, out-of-range terms = 0)
    ml = m & (offw[None, :] > 0)
    mt = m & (offh[:, None] > 0)
    px1 = tl.load(px_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    px1l = tl.load(px_ptr + b2 + idx - 1, mask=ml, other=0.0).to(tl.float32)
    py1 = tl.load(py_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    py1t = tl.load(py_ptr + b2 + idx - W, mask=mt, other=0.0).to(tl.float32)
    px2 = tl.load(px_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    px2l = tl.load(px_ptr + b2 + HW + idx - 1, mask=ml, other=0.0).to(tl.float32)
    py2 = tl.load(py_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    py2t = tl.load(py_ptr + b2 + HW + idx - W, mask=mt, other=0.0).to(tl.float32)

    # ---- primal update (grouped exactly like _divergence's d1 + d2, so the
    # only float reassociation vs. the eager reference is torch.hypot below)
    un1 = u1 + d1 + theta * ((px1 - px1l) + (py1 - py1t))
    un2 = u2 + d2 + theta * ((px2 - px2l) + (py2 - py2t))

    if CALC_ERR:
        e = (un1 - u1) * (un1 - u1) + (un2 - u2) * (un2 - u2)
        s = tl.sum(tl.where(m, e, 0.0))
        tl.store(errp_ptr + b * NT + pid_h * NW + pid_w, s)

    dt = u_ptr.dtype.element_ty
    tl.store(u_ptr + b2 + idx, tl.where(act, un1, u1).to(dt), mask=m)
    tl.store(u_ptr + b2 + HW + idx, tl.where(act, un2, u2).to(dt), mask=m)


# --------------------------------------------------------------------------- #
# Kernel 2: forward gradient + dual ascent + projection -- in place
#           (+ the error fold / freeze-mask update, on tile 0)
# --------------------------------------------------------------------------- #
@triton.jit
def _estimate_dual_kernel(
    u_ptr,            # (B, 2, H, W) in
    px_ptr, py_ptr,   # (B, 2, H, W) in/out
    active_ptr,       # (B,) bool  in
    active_out_ptr,   # (B,) bool out (CALC_ERR only; may alias active_ptr)
    errp_ptr,         # (B, NT) fp32 in  (CALC_ERR only)
    err_ptr,          # (B,) fp32 out    (CALC_ERR only)
    H, W, HW, CHW,
    NW, NT,
    taut, scaled_eps,
    BH: tl.constexpr, BW: tl.constexpr, CALC_ERR: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = tl.program_id(2)

    offh = pid_h * BH + tl.arange(0, BH)
    offw = pid_w * BW + tl.arange(0, BW)
    hin = offh[:, None] < H
    win = offw[None, :] < W
    m = hin & win
    idx = offh[:, None] * W + offw[None, :]
    b2 = b * CHW

    act = tl.load(active_ptr + b).to(tl.int1)

    hasr = win & (offw[None, :] + 1 < W)
    hasd = hin & (offh[:, None] + 1 < H)
    mr = m & hasr
    md = m & hasd

    u1 = tl.load(u_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    u2 = tl.load(u_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    u1r = tl.load(u_ptr + b2 + idx + 1, mask=mr, other=0.0).to(tl.float32)
    u1d = tl.load(u_ptr + b2 + idx + W, mask=md, other=0.0).to(tl.float32)
    u2r = tl.load(u_ptr + b2 + HW + idx + 1, mask=mr, other=0.0).to(tl.float32)
    u2d = tl.load(u_ptr + b2 + HW + idx + W, mask=md, other=0.0).to(tl.float32)

    ux1 = tl.where(hasr, u1r - u1, 0.0)
    uy1 = tl.where(hasd, u1d - u1, 0.0)
    ux2 = tl.where(hasr, u2r - u2, 0.0)
    uy2 = tl.where(hasd, u2d - u2, 0.0)

    ng1 = 1.0 + taut * tl.sqrt(ux1 * ux1 + uy1 * uy1)
    ng2 = 1.0 + taut * tl.sqrt(ux2 * ux2 + uy2 * uy2)

    px1 = tl.load(px_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    py1 = tl.load(py_ptr + b2 + idx, mask=m, other=0.0).to(tl.float32)
    px2 = tl.load(px_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)
    py2 = tl.load(py_ptr + b2 + HW + idx, mask=m, other=0.0).to(tl.float32)

    dt = px_ptr.dtype.element_ty
    tl.store(px_ptr + b2 + idx, tl.where(act, (px1 + taut * ux1) / ng1, px1).to(dt), mask=m)
    tl.store(py_ptr + b2 + idx, tl.where(act, (py1 + taut * uy1) / ng1, py1).to(dt), mask=m)
    tl.store(px_ptr + b2 + HW + idx, tl.where(act, (px2 + taut * ux2) / ng2, px2).to(dt), mask=m)
    tl.store(py_ptr + b2 + HW + idx, tl.where(act, (py2 + taut * uy2) / ng2, py2).to(dt), mask=m)

    # ---- convergence fold: the (tile 0, b) program folds sample b's partials.
    # Vector accumulator + one final tree reduction => the summation order is a
    # function of (NT, RBLOCK) only, never of B or of scheduling.
    if CALC_ERR:
        if pid_h + pid_w == 0:
            acc = tl.zeros((RBLOCK,), dtype=tl.float32)
            for k in range(0, NT, RBLOCK):
                o = k + tl.arange(0, RBLOCK)
                acc += tl.load(errp_ptr + b * NT + o, mask=o < NT, other=0.0)
            e = tl.sum(acc, axis=0)
            # frozen samples must report 0 so the batch max tracks the
            # still-active ones only (torch_flow._tvl1_step)
            e = tl.where(act, e, 0.0)
            tl.store(err_ptr + b, e)
            tl.store(active_out_ptr + b, (act & (e > scaled_eps)).to(tl.int1))


# --------------------------------------------------------------------------- #
# Host-side driver: state cache + config selection
# --------------------------------------------------------------------------- #
class _State:
    """Per-(B, H, W, dtype) launch state: tiling, grid, scratch planes."""

    __slots__ = ("BH", "BW", "warps", "RBLOCK", "grid", "shape_args",
                 "errp", "err", "alt", "cur")


def _launch_pair(st, u, px, py, i1w, rho_c, thr, pos, sg, active,
                 l_t, theta, taut, scaled_eps, calc_error, active_out=None):
    """The two launches that make up one primal-dual iteration."""
    h, w, hw, chw, nw, nt = st.shape_args
    _estimate_u_kernel[st.grid](
        u, px, py, i1w, rho_c, thr, pos, sg, active, st.errp,
        h, w, hw, chw, nw, nt, l_t, theta,
        BH=st.BH, BW=st.BW, CALC_ERR=calc_error, num_warps=st.warps,
    )
    _estimate_dual_kernel[st.grid](
        u, px, py, active, active if active_out is None else active_out,
        st.errp, st.err,
        h, w, hw, chw, nw, nt, taut, scaled_eps,
        BH=st.BH, BW=st.BW, CALC_ERR=calc_error, RBLOCK=st.RBLOCK,
        num_warps=st.warps,
    )


class TritonTVL1Step:
    """Drop-in replacement for ``torch_flow._tvl1_step`` backed by the two kernels.

    Signature and return tuple match ``_tvl1_step`` exactly, with two
    documented differences:

    * ``u``, ``px`` and ``py`` are updated **in place** and returned as the same
      tensor objects (the caller's ``u, ..., px, py, ... = step(...)`` rebinding
      is therefore a no-op, which is what makes the in-place form transparent);
    * the returned ``err_max`` is the per-sample error vector ``(B,)``, not a
      scalar -- the batch max is left to the host, which only needs it on the
      throttled sync iterations.  ``torch_flow`` reduces it with ``_err_scalar``.
    """

    def __init__(self, autotune: bool | None = None):
        self._cache: dict[tuple, dict] = {}   # (B, H, W, dtype, dev) -> launch state
        self._cfg: dict[tuple, tuple] = {}    # (H, W, dtype, dev)    -> (BH, BW, warps)
        if autotune is None:
            autotune = os.environ.get("TVL1_TRITON_AUTOTUNE", "1") != "0"
        self.autotune = autotune

    # -- config selection ---------------------------------------------------
    def _pick_config(self, ref: tuple) -> tuple[int, int, int]:
        """Time `_CONFIGS` on the actual planes, once per (H, W, dtype).

        Deliberately *not* keyed on B: the number of tiles per sample (and hence
        the summation order of the convergence reduction) must be a function of
        the image shape alone, or the freeze schedule -- and with it the result --
        would depend on the batch size.  The cost is that a solver instance tunes
        at whatever B it sees first; the benchmark builds a fresh instance per
        cell so each B gets its own tuning.
        """
        forced = os.environ.get("TVL1_TRITON_BLOCK")
        if forced:
            bh, bw, nw = (int(v) for v in forced.split(","))
            return bh, bw, nw
        if not self.autotune:
            return _CONFIGS[1]
        best, best_t = _CONFIGS[1], float("inf")
        for cfg in _CONFIGS:
            try:
                t = self._time_config(cfg, ref)
            except Exception:  # noqa: BLE001  (compile failure on this shape)
                continue
            if t < best_t:
                best, best_t = cfg, t
        return best

    def _time_config(self, cfg, ref) -> float:
        st = self._make_state(cfg, ref)
        for _ in range(3):
            _launch_pair(st, *ref, True)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(20):
            _launch_pair(st, *ref, True)
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1)

    def _make_state(self, cfg, ref) -> _State:
        bh, bw, nwarps = cfg
        u = ref[0]
        b, _, h, w = u.shape
        nw, nh = triton.cdiv(w, bw), triton.cdiv(h, bh)
        nt = nw * nh
        st = _State()
        st.BH, st.BW, st.warps = bh, bw, nwarps
        st.RBLOCK = min(256, max(32, triton.next_power_of_2(nt)))
        st.grid = (nw, nh, b)
        st.shape_args = (h, w, h * w, 2 * h * w, nw, nt)
        st.errp = torch.empty(b, nt, dtype=torch.float32, device=u.device)
        st.err = torch.empty(b, dtype=torch.float32, device=u.device)
        st.alt = torch.empty(b, 1, 1, 1, dtype=torch.bool, device=u.device)
        st.cur = None
        return st

    def _setup(self, key, ref) -> _State:
        u, px, py, i1w, rho_c, thr, pos, sg, active = ref[:9]
        for name, t in (("u", u), ("px", px), ("py", py), ("i1wxy", i1w),
                        ("rho_c", rho_c), ("thr", thr), ("pos", pos),
                        ("safe_grad", sg), ("active", active)):
            if not t.is_contiguous():
                raise RuntimeError(f"backend='triton' needs a contiguous {name}")
        ckey = key[1:]
        cfg = self._cfg.get(ckey)
        if cfg is None:
            # Autotuning mutates u/px/py/active; snapshot and restore so the
            # first iteration of a level is not silently advanced by timing runs.
            snap = [t.clone() for t in (u, px, py, active)]
            cfg = self._pick_config(ref)
            for dst, src in zip((u, px, py, active), snap):
                dst.copy_(src)
            self._cfg[ckey] = cfg
        st = self._make_state(cfg, ref)
        self._cache[key] = st
        return st

    # -- the _tvl1_step-compatible entry point ------------------------------
    # Kept deliberately lean: it runs ~7 500 times per solve, so every avoidable
    # dict build / dtype coercion here shows up in the wall clock.
    def __call__(
        self, u, u3, px, py, p3x, p3y, active,
        rho_c, i1wxy, thr, pos, safe_grad,
        l_t, theta, taut, gamma, scaled_eps, calc_error,
    ):
        sh = u.shape
        key = (sh[0], sh[2], sh[3], u.dtype, u.get_device())
        st = self._cache.get(key)
        ref = (u, px, py, i1wxy, rho_c, thr, pos, safe_grad, active,
               l_t, theta, taut, scaled_eps)
        if st is None:
            if gamma != 0.0:
                raise NotImplementedError("backend='triton' does not implement gamma != 0")
            st = self._setup(key, ref)
        elif not u.is_contiguous():
            # u is rebound by the median filter between outer iterations
            raise RuntimeError("backend='triton' needs a contiguous u (updated in place)")

        if calc_error:
            out = st.alt if active is not st.alt else st.cur
            if out is None or out is active:
                out = torch.empty_like(active)
            st.cur, st.alt = active, out
            _launch_pair(st, *ref, True, out)
            return u, None, px, py, None, None, out, st.err
        _launch_pair(st, *ref, False)
        return u, None, px, py, None, None, active, None

    # -- diagnostics --------------------------------------------------------
    def config_report(self) -> list[str]:
        return [
            f"B={k[0]} {k[1]}x{k[2]} {str(k[3]).replace('torch.', '')}: "
            f"BH={v.BH} BW={v.BW} warps={v.warps} tiles={v.shape_args[5]} grid={v.grid}"
            for k, v in sorted(self._cache.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        ]
