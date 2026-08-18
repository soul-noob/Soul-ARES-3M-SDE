"""ARES-DiT Adaptive Flow sampler for ComfyUI.

Designed for Flow/Rectified-Flow image DiTs / MMDiTs such as Anima,
Krea 2 and Flux-family models.

Normal path:
    Heun bootstrap -> variable-step AB2/AB3 -> adaptive AB fallback

Repair path when AB2/AB3 disagree too much:
    rewind in Flow time -> RK2/Heun refinement -> optional Flow-consistent
    stochastic restart -> re-integrate to the same boundary -> re-test.

This deliberately contains NO EDM/VE SDE logic.

The model callable is assumed to be ComfyUI's wrapped model which returns
x0/denoised. For the standard Flow/CONST path:
    x0_hat = x - t * v
so:
    v = (x - x0_hat) / t
"""

from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from tqdm.auto import trange
except Exception:  # pragma: no cover
    def trange(n, disable=False):
        return range(n)

History = List[Tuple[float, Tensor]]


def _scalar(x: Tensor) -> float:
    return float(x.detach().item())


def _view(t: Tensor, ndim: int) -> Tensor:
    while t.ndim < ndim:
        t = t.unsqueeze(-1)
    return t


def _safe_t(t: Tensor, dtype: torch.dtype) -> Tensor:
    return torch.clamp(t, min=torch.finfo(dtype).eps)


def _rms(x: Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt().item())


def _disagreement(a: Tensor, b: Tensor, base: Tensor) -> float:
    return _rms(a - b) / (_rms(base) + 1e-8)


def _weights(nodes: List[float], start: float, end: float) -> List[float]:
    """Integral weights of the Lagrange interpolant on [start,end]."""
    n = len(nodes)
    if n == 1:
        return [end - start]
    out: List[float] = []
    for j, nj in enumerate(nodes):
        poly = [1.0]
        den = 1.0
        for k, nk in enumerate(nodes):
            if k == j:
                continue
            den *= nj - nk
            nxt = [0.0] * (len(poly) + 1)
            for p, c in enumerate(poly):
                nxt[p] += -nk * c
                nxt[p + 1] += c
            poly = nxt
        integ = 0.0
        for p, c in enumerate(poly):
            integ += c / (p + 1) * (end ** (p + 1) - start ** (p + 1))
        out.append(integ / den)
    return out


def _predict(x: Tensor, history: History, t: float, t_next: float, order: int):
    n = min(order, len(history), 3)
    if n < 1:
        raise ValueError("No derivative history")
    h = history[:n]
    ws = _weights([a for a, _ in h], t, t_next)
    delta = torch.zeros_like(x)
    for w, (_, v) in zip(ws, h):
        delta.add_(v, alpha=w)
    return x + delta, ws


def _flow_velocity(model, x: Tensor, t: Tensor, s_in: Tensor, extra_args: Dict):
    ts = _safe_t(t, x.dtype)
    den = model(x, ts * s_in, **extra_args)
    return (x - den) / _view(ts, x.ndim), den


def _heun_step(model, x, t, t_next, s_in, extra_args, v0=None):
    h = t_next - t
    if v0 is None:
        v0, _ = _flow_velocity(model, x, t, s_in, extra_args)
    xp = x + h * v0
    vp, _ = _flow_velocity(model, xp, t_next, s_in, extra_args)
    xc = x + 0.5 * h * (v0 + vp)
    ve, _ = _flow_velocity(model, xc, t_next, s_in, extra_args)
    return xc, ve, 3


def _rk2_step(model, x, t, t_next, s_in, extra_args):
    h = t_next - t
    k1, _ = _flow_velocity(model, x, t, s_in, extra_args)
    km, _ = _flow_velocity(model, x + 0.5 * h * k1, t + 0.5 * h, s_in, extra_args)
    xn = x + h * km
    ve, _ = _flow_velocity(model, xn, t_next, s_in, extra_args)
    return xn, ve, 3


def _repair_integrate(model, x0, t0, t1, s_in, extra_args, method, subdivisions):
    x = x0
    h = (t1 - t0) / int(subdivisions)
    t = t0
    hist: History = []
    nfe = 0
    for _ in range(int(subdivisions)):
        tn = t + h
        if method == "heun":
            x, v, nf = _heun_step(model, x, t, tn, s_in, extra_args)
        elif method == "rk2":
            x, v, nf = _rk2_step(model, x, t, tn, s_in, extra_args)
        else:
            raise ValueError(method)
        hist.insert(0, (_scalar(tn), v))
        hist[:] = hist[:3]
        nfe += nf
        t = tn
    return x, hist, nfe


def _estimate_eps(x: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    ts = _view(_safe_t(t, x.dtype), x.ndim)
    return (x - (1.0 - ts) * x0) / ts


def _restart_flow_state(x, x0, t_current, t_rewind, rho, eps_new):
    eps_hat = _estimate_eps(x, x0, t_current)
    rho = max(0.0, min(float(rho), 0.999))
    eps_r = math.sqrt(max(0.0, 1.0 - rho * rho)) * eps_hat + rho * eps_new
    tr = _view(t_rewind, x.ndim)
    return (1.0 - tr) * x0 + tr * eps_r


class _Noise:
    def __init__(self, x: Tensor, seed: Optional[int]):
        self.device = x.device
        self.base = None if seed is None else int(seed)
        self.i = 0
    def __call__(self, like: Tensor):
        if self.base is None:
            return torch.randn_like(like)
        seed = (self.base + 104729 * self.i + 1723) % (2**63 - 1)
        self.i += 1
        g = torch.Generator(device=like.device)
        g.manual_seed(seed)
        return torch.randn(like.shape, device=like.device, dtype=like.dtype, generator=g)


@torch.no_grad()
def sample_ares_dit_adaptive(
    model,
    x: Tensor,
    sigmas: Tensor,
    extra_args: Optional[Dict] = None,
    callback: Optional[Callable] = None,
    disable: Optional[bool] = None,
    max_order: int = 3,
    accept_threshold: float = 0.004,
    repair_threshold: float = 0.010,
    severe_threshold: float = 0.025,
    min_rewind_steps: int = 1,
    max_rewind_steps: int = 3,
    max_repair_attempts: int = 3,
    max_repair_nfe: int = 10,
    repair_subdivisions: int = 1,
    max_repair_subdivisions: int = 4,
    stochastic_repair: bool = True,
    noise_min_rho: float = 0.03,
    noise_max_rho: float = 0.18,
    stochastic_after_attempt: int = 2,
    min_stochastic_time: float = 0.08,
    tail_start: float = 0.18,
    tail_order: int = 2,
    max_weight_abs: float = 12.0,
    debug: bool = False,
):
    """ComfyUI-compatible adaptive ARES-DiT Flow sampler."""
    extra_args = {} if extra_args is None else extra_args
    if sigmas.numel() < 2:
        return x

    s_in = x.new_ones([x.shape[0]])
    noise = _Noise(x, extra_args.get("seed"))
    history: History = []
    repairs = 0
    repair_nfe_total = 0
    n_steps = int(sigmas.numel() - 1)

    # ---------------------------------------------------------------
    # Heun bootstrap. Store v(t0) and a fresh v(t1) at the corrected
    # endpoint so the next step can immediately use AB2.
    # ---------------------------------------------------------------
    t0, t1 = sigmas[0], sigmas[1]
    v0, den0 = _flow_velocity(model, x, t0, s_in, extra_args)
    xp = x + (t1 - t0) * v0
    vp, _ = _flow_velocity(model, xp, t1, s_in, extra_args)
    x = x + 0.5 * (t1 - t0) * (v0 + vp)
    v1, _ = _flow_velocity(model, x, t1, s_in, extra_args)
    history = [(_scalar(t1), v1), (_scalar(t0), v0)]
    if callback is not None:
        callback({"x": x, "i": 0, "sigma": t0, "sigma_hat": t0, "denoised": den0})

    for i in trange(1, n_steps, disable=disable):
        t = sigmas[i]
        t_next = sigmas[i + 1]

        # Fresh derivative at the actual current state.
        v, x0_hat = _flow_velocity(model, x, t, s_in, extra_args)
        t_value = _scalar(t)
        if history and abs(history[0][0] - t_value) < 1e-12:
            history[0] = (t_value, v)
        else:
            history.insert(0, (t_value, v))
        history[:] = history[:3]

        near_tail = _scalar(t) <= float(tail_start)
        desired = min(max_order, len(history))
        if near_tail:
            desired = min(desired, int(tail_order))

        x1, w1 = _predict(x, history, _scalar(t), _scalar(t_next), 1)
        x2, w2 = _predict(x, history, _scalar(t), _scalar(t_next), min(2, desired))

        use_ab3 = desired >= 3
        if use_ab3:
            x3, w3 = _predict(x, history, _scalar(t), _scalar(t_next), 3)
            if max(abs(z) for z in w3) > float(max_weight_abs):
                use_ab3 = False
                w3 = w2
                x3 = x2
        else:
            x3, w3 = x2, w2

        err = _disagreement(x3, x2, x) if use_ab3 else 0.0

        if use_ab3 and err < accept_threshold:
            x = x3
            order = 3
        elif desired >= 2 and err < repair_threshold:
            x = x2
            order = 2
        else:
            # -------------------------------------------------------
            # Repair. The boundary is the CURRENT time t. We rewind to
            # a previous schedule point, optionally perturb it in Flow
            # space, then numerically return to t. Only after returning
            # do we recompute AB2/AB3 for t -> t_next.
            # -------------------------------------------------------
            original_x = x
            original_history = list(history)
            original_err = err
            sev = max(0.0, min(1.0, (err - repair_threshold) / max(severe_threshold - repair_threshold, 1e-8)))
            rewind_steps = int(round(min_rewind_steps + sev * (max_rewind_steps - min_rewind_steps)))
            rewind_steps = max(min_rewind_steps, min(max_rewind_steps, rewind_steps, i))
            rewind_idx = max(0, i - rewind_steps)
            t_rewind = sigmas[rewind_idx]

            allow_noise = stochastic_repair and _scalar(t) >= float(min_stochastic_time)
            best = None
            best_score = float("inf")
            best_hist = None
            used_nfe = 0
            subdivisions = max(1, int(repair_subdivisions))

            for attempt in range(int(max_repair_attempts)):
                method = "heun" if attempt % 2 == 0 else "rk2"
                use_noise = allow_noise and attempt >= int(stochastic_after_attempt)

                if use_noise:
                    q = max(0.0, min(1.0, (original_err - repair_threshold) / max(severe_threshold - repair_threshold, 1e-8)))
                    rho = noise_min_rho + q * (noise_max_rho - noise_min_rho)
                    xr = _restart_flow_state(original_x, x0_hat, t, t_rewind, rho, noise(original_x))
                else:
                    # Deterministic flow rewind: stay on the model-implied
                    # linear interpolation between x0_hat and eps_hat.
                    eps_hat = _estimate_eps(original_x, x0_hat, t)
                    tr = _view(t_rewind, original_x.ndim)
                    xr = (1.0 - tr) * x0_hat + tr * eps_hat

                trial_x, trial_hist, nf = _repair_integrate(
                    model, xr, t_rewind, t, s_in, extra_args, method, subdivisions
                )
                used_nfe += nf
                if used_nfe > int(max_repair_nfe):
                    break
                if len(trial_hist) < 2:
                    continue

                trial2, _ = _predict(trial_x, trial_hist, _scalar(t), _scalar(t_next), 2)
                if len(trial_hist) >= 3:
                    trial3, _ = _predict(trial_x, trial_hist, _scalar(t), _scalar(t_next), 3)
                    trial_err = _disagreement(trial3, trial2, trial_x)
                else:
                    trial3, trial_err = trial2, float("inf")

                drift = _rms(trial_x - original_x) / (_rms(original_x) + 1e-8)
                score = trial_err + 0.10 * max(0.0, drift - 0.10)
                if score < best_score:
                    best_score = score
                    best = trial_x
                    best_hist = trial_hist

                if trial_err < accept_threshold:
                    break
                if attempt >= 1 and subdivisions < max_repair_subdivisions:
                    subdivisions = min(max_repair_subdivisions, subdivisions * 2)

            if best is not None and best_hist is not None and best_score < original_err:
                x = best
                history = best_hist[:3]
                repaired2, _ = _predict(x, history, _scalar(t), _scalar(t_next), min(2, len(history)))
                if len(history) >= 3:
                    repaired3, repaired_err = _predict(x, history, _scalar(t), _scalar(t_next), 3)
                    repaired_err = _disagreement(repaired3, repaired2, x)
                else:
                    repaired3, repaired_err = repaired2, float("inf")

                if len(history) >= 3 and repaired_err < accept_threshold:
                    x = repaired3
                    order = 3
                else:
                    x = repaired2
                    order = 2
                repairs += 1
                repair_nfe_total += used_nfe
            else:
                # Safety valve: never amplify the image just because the
                # repair system got excited. Fall back to conservative AB2.
                x = x2
                history = original_history
                order = 2

        if callback is not None:
            payload = {
                "x": x,
                "i": i,
                "sigma": t,
                "sigma_hat": t,
                "denoised": x0_hat,
                "ares_order": order,
                "ares_disagreement": err,
                "ares_repairs": repairs,
                "ares_repair_nfe": repair_nfe_total,
            }
            if debug:
                payload["ares_ab2"] = x2
                payload["ares_ab3"] = x3
            callback(payload)

    return x


# Backwards-compatible name for the existing registration in the repo.
sample_restart_er_3ma = sample_ares_dit_adaptive
