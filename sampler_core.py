# ======================================================================
# SOUL ARES 3M SDE
#
# Restart + ER-SDE + 3rd-Order Multistep + aRK4 + Midpoint Bootstrap
#
# Architecture:
#
#   Phase 1  : 2-NFE midpoint bootstrap
#   Phase 2  : variable-step 3rd-order multistep cruising
#   Phase 3  : restart macro-noise
#   Phase 4  : targeted aRK4 recovery
#   Phase 5  : variance-locked ER-SDE micro-noise
#
# The multistep engine follows the ARES mathematical design:
#
#   d_pred = polynomial extrapolation of recent derivatives
#
# but computes the integration coefficients for the ACTUAL sigma
# locations instead of assuming equally-spaced steps.
# ======================================================================

import math

import torch
from torch import Tensor
from tqdm.auto import trange


# ======================================================================
# Utility functions
# ======================================================================

def _scalar_sigma(value: Tensor) -> float:
    """
    Convert a scalar tensor to a Python float.

    Sigma values supplied by ComfyUI are normally scalar tensors.
    Keeping the schedule geometry in Python floats makes the polynomial
    coefficient calculations simple and avoids unnecessary tensor work.
    """
    return float(value.detach().item())


def _safe_sigma(value: Tensor, dtype: torch.dtype) -> Tensor:
    """
    Prevent division by zero when evaluating the model at sigma ~= 0.
    """
    tiny = torch.finfo(dtype).tiny
    return torch.clamp(value, min=tiny)


def _derivative(
    model,
    x: Tensor,
    sigma: Tensor,
    s_in: Tensor,
    extra_args: dict,
):
    """
    Convert the model's denoised prediction into the diffusion ODE
    derivative used by the ARES multistep integrator.

        d = (x - denoised) / sigma

    Sigma is broadcast across the latent dimensions.
    """
    sigma_safe = _safe_sigma(sigma, x.dtype)

    denoised = model(
        x,
        sigma_safe * s_in,
        **extra_args,
    )

    sigma_view = sigma_safe

    while sigma_view.ndim < x.ndim:
        sigma_view = sigma_view.unsqueeze(-1)

    return (x - denoised) / sigma_view, denoised


# ======================================================================
# Variable-step polynomial integration
# ======================================================================

def _lagrange_integral_weights(
    nodes: list[float],
    start: float,
    end: float,
) -> list[float]:
    """
    Integrate the Lagrange interpolation polynomial through `nodes`
    from `start` to `end`.

    If:

        d(sigma) ~= P(sigma)

    then:

        integral(start -> end) P(sigma) d_sigma

    gives the multistep update.

    This is the key difference from the old implementation.

    The old implementation used:

        3/2, -1/2

    and:

        23/12, -16/12, 5/12

    regardless of the actual sigma spacing.

    Those coefficients are only valid for equally spaced nodes.

    A diffusion sigma schedule is generally NOT equally spaced, and ARES
    additionally modifies the effective deterministic endpoint through
    ER-SDE. Therefore the coefficients must be recomputed for the actual
    history geometry.
    """
    count = len(nodes)

    if count == 1:
        return [end - start]

    weights = []

    for j in range(count):
        # Polynomial coefficients in ascending powers:
        #
        # p(x) = c0 + c1*x + c2*x^2 + ...
        polynomial = [1.0]
        denominator = 1.0

        node_j = nodes[j]

        for k in range(count):
            if k == j:
                continue

            node_k = nodes[k]

            denominator *= node_j - node_k

            # Multiply polynomial by:
            #
            #     (x - node_k)
            #
            new_poly = [0.0] * (len(polynomial) + 1)

            for power, coefficient in enumerate(polynomial):
                new_poly[power] += -node_k * coefficient
                new_poly[power + 1] += coefficient

            polynomial = new_poly

        # Integrate the polynomial from start to end.
        integral = 0.0

        for power, coefficient in enumerate(polynomial):
            integral += (
                coefficient
                / float(power + 1)
                * (
                    end ** (power + 1)
                    - start ** (power + 1)
                )
            )

        weights.append(integral / denominator)

    return weights


def _multistep_predict(
    history: list[tuple[float, Tensor]],
    sigma: float,
    sigma_target: float,
) -> Tensor:
    """
    Perform variable-step Adams-Bashforth-style polynomial integration.

    History is stored newest-first:

        history[0] = derivative at current sigma
        history[1] = derivative at previous sigma
        history[2] = derivative at previous-previous sigma

    The interpolation polynomial is integrated directly from the current
    sigma to the target sigma.

    This gives:

        AB1  -> 1 history point
        AB2  -> 2 history points
        AB3  -> 3 history points

    while remaining valid for non-uniform sigma spacing.
    """

    count = min(len(history), 3)

    selected = history[:count]

    nodes = [
        item[0]
        for item in selected
    ]

    derivatives = [
        item[1]
        for item in selected
    ]

    weights = _lagrange_integral_weights(
        nodes=nodes,
        start=sigma,
        end=sigma_target,
    )

    result = torch.zeros_like(derivatives[0])

    for weight, derivative in zip(weights, derivatives):
        result = result + derivative * weight

    return result


# ======================================================================
# Restart schedule construction
# ======================================================================

def _build_restart_schedule(
    sigmas: Tensor,
) -> tuple[Tensor, set[int]]:
    """
    Construct ARES restart jumps from the ORIGINAL sigma schedule.

    A restart at index k creates:

        sigma[k]
        sigma[k-2]
        sigma[k-1]
        sigma[k]
        sigma[k+1]

    The backward jump therefore becomes:

        sigma[k] -> sigma[k-2]

    followed by two ordinary forward-denoising steps back to sigma[k].

    IMPORTANT:
    Restart positions are calculated from the original schedule and
    inserted in a single pass. This prevents the second insertion from
    shifting the first insertion's location.
    """

    original = sigmas.detach().clone()

    steps = original.numel() - 1

    if steps <= 0:
        return original, set()

    values = [
        original[i]
        for i in range(original.numel())
    ]

    # Only inject ARES restarts into an ordinary monotonically decreasing
    # schedule. If the caller has already supplied a schedule containing
    # upward jumps, leave it untouched.
    decreasing = all(
        values[i] >= values[i + 1]
        for i in range(len(values) - 1)
    )

    if not decreasing:
        return original, set()

    restart_indices: list[int] = []

    if 15 <= steps <= 25:
        # One restart around the middle.
        k = steps // 2

        if k >= 3:
            restart_indices.append(k)

    elif 26 <= steps <= 40:
        # Two restarts around 1/3 and 2/3.
        k1 = steps // 3
        k2 = (steps * 2) // 3

        if k1 >= 3 and k2 >= 3:
            restart_indices.extend([k1, k2])

    if not restart_indices:
        return original, set()

    restart_indices = sorted(set(restart_indices))

    result: list[Tensor] = []

    for i in range(steps):
        result.append(values[i])

        if i in restart_indices:
            # The jump itself:
            #
            # sigma[i] -> sigma[i-2]
            #
            # followed by the original trajectory:
            #
            # sigma[i-2] -> sigma[i-1] -> sigma[i]
            #
            result.extend(
                values[i - 2:i + 1]
            )

    # Final sigma.
    result.append(values[-1])

    new_sigmas = torch.stack(result).to(
        device=sigmas.device,
        dtype=sigmas.dtype,
    )

    # Restart positions are represented by the position in the newly
    # constructed schedule where:
    #
    #     new_sigmas[position + 1] > new_sigmas[position]
    #
    # We identify them directly rather than trying to translate the
    # original indices after insertion.
    jump_positions: set[int] = set()

    for i in range(new_sigmas.numel() - 1):
        if new_sigmas[i + 1] > new_sigmas[i]:
            jump_positions.add(i)

    return new_sigmas, jump_positions


# ======================================================================
# Main sampler
# ======================================================================

@torch.no_grad()
def sample_restart_er_3ma(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    gamma=0.15,
    s_noise=1.0,
    noise_sampler=None,
):
    """
    Soul ARES 3M SDE.

    Phase 1:
        Midpoint bootstrap.
        2 NFEs.

    Phase 2:
        Variable-step third-order polynomial multistep cruising.
        1 NFE per step after initialization.

    Phase 3:
        Restart macro-noise.
        No NFE.

    Phase 4:
        aRK4 recovery immediately after restart.
        4 NFEs.

    Phase 5:
        ER-SDE micro-noise with variance lock.

    Parameters
    ----------
    eta:
        ER-SDE noise strength.

    gamma:
        Maximum ER-SDE noise ratio relative to sigma_next.

    s_noise:
        Global noise multiplier supplied by ComfyUI.

    noise_sampler:
        ComfyUI/k-diffusion noise sampler used to preserve seeded
        stochastic behavior.
    """

    extra_args = {} if extra_args is None else extra_args

    # ComfyUI expects sigma to be supplied once per batch element.
    s_in = x.new_ones([x.shape[0]])

    # --------------------------------------------------------------
    # Build the ARES restart schedule.
    # --------------------------------------------------------------

    sigmas, restart_positions = _build_restart_schedule(sigmas)

    # --------------------------------------------------------------
    # State.
    #
    # Each history entry is:
    #
    #     (sigma_at_derivative, derivative_tensor)
    #
    # Newest derivative is stored first.
    # --------------------------------------------------------------

    history: list[tuple[float, Tensor]] = []

    # Set after a backward macro-noise jump.
    just_restarted = False

    # --------------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------------

    for i in trange(
        len(sigmas) - 1,
        disable=disable,
    ):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_value = _scalar_sigma(sigma)
        sigma_next_value = _scalar_sigma(sigma_next)

        # ==========================================================
        # PHASE 3
        #
        # RESTART MACRO-NOISE
        #
        # sigma_next > sigma means the schedule is intentionally
        # travelling backward toward a noisier state.
        #
        # Variance addition:
        #
        #   sigma_next^2
        #       =
        #   sigma^2 + sigma_jump^2
        #
        # therefore:
        #
        #   sigma_jump = sqrt(sigma_next^2 - sigma^2)
        # ==========================================================

        if sigma_next_value > sigma_value:
            variance_gap = (
                sigma_next.square()
                - sigma.square()
            )

            variance_gap = torch.clamp(
                variance_gap,
                min=0.0,
            )

            sigma_jump = torch.sqrt(
                variance_gap
            )

            if noise_sampler is not None:
                epsilon = noise_sampler(
                    sigma,
                    sigma_next,
                )
            else:
                epsilon = torch.randn_like(x)

            x = (
                x
                + epsilon
                * sigma_jump
                * float(s_noise)
            )

            # The old trajectory is no longer mathematically valid.
            history.clear()

            just_restarted = True

            # A restart itself does not evaluate the model.
            #
            # We deliberately do not send a fake "denoised=x"
            # callback because x is NOT a denoised prediction.
            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": x,
                })

            continue

        # ==========================================================
        # PHASE 5
        #
        # ER-SDE TARGET
        #
        # sigma_up is constrained by:
        #
        #   min(
        #       ER_Target,
        #       gamma * sigma_next
        #   )
        #
        # sigma_down is then chosen so the final variance remains
        # sigma_next^2.
        # ==========================================================

        if sigma_next_value > 0.0 and sigma_value > 0.0:

            ratio = sigma_next / sigma

            ratio_sq = torch.clamp(
                1.0 - ratio.square(),
                min=0.0,
            )

            er_target = (
                float(eta)
                * sigma_next
                * torch.sqrt(ratio_sq)
            )

            gamma_target = (
                float(gamma)
                * sigma_next
            )

            sigma_up = torch.minimum(
                er_target,
                gamma_target,
            )

            sigma_up = torch.clamp(
                sigma_up,
                min=0.0,
            )

            sigma_down_sq = (
                sigma_next.square()
                - sigma_up.square()
            )

            sigma_down = torch.sqrt(
                torch.clamp(
                    sigma_down_sq,
                    min=0.0,
                )
            )

        else:
            sigma_up = torch.zeros_like(sigma)
            sigma_down = torch.zeros_like(sigma)

        sigma_down_value = _scalar_sigma(
            sigma_down
        )

        # ==========================================================
        # PHASE 1
        #
        # MIDPOINT BOOTSTRAP
        #
        # 2 NFEs.
        #
        # IMPORTANT:
        #
        # We do NOT insert d_mid into the multistep history.
        #
        # d_mid is evaluated at sigma_mid, while the resulting noisy
        # state belongs to sigma_next. Treating d_mid as though it
        # were a derivative evaluated at sigma_next corrupts the
        # history geometry.
        #
        # The following step therefore starts cleanly with AB1.
        # ==========================================================

        if i == 0 and not history and not just_restarted:

            # First derivative.
            d_0, denoised_0 = _derivative(
                model,
                x,
                sigma,
                s_in,
                extra_args,
            )

            # Exact midpoint of the deterministic integration interval.
            sigma_mid = (
                sigma
                + sigma_down
            ) * 0.5

            sigma_mid_value = _scalar_sigma(
                sigma_mid
            )

            # Half-step prediction.
            x_mid = (
                x
                + d_0
                * (sigma_mid - sigma)
            )

            # Second NFE.
            d_mid, denoised_mid = _derivative(
                model,
                x_mid,
                sigma_mid,
                s_in,
                extra_args,
            )

            # Full midpoint update.
            x_next_det = (
                x
                + d_mid
                * (sigma_down - sigma)
            )

            # No history seed here.
            #
            # The derivative is located at sigma_mid, not sigma_down,
            # and after ER-SDE noise x will belong to sigma_next.
            history.clear()

            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised_0,
                })

        # ==========================================================
        # PHASE 4
        #
        # TARGETED aRK4 RECOVERY
        #
        # 4 NFEs after a restart.
        #
        # This is deliberately performed on the deterministic portion
        # of the step.
        # ==========================================================

        elif just_restarted:

            h = (
                sigma_down
                - sigma
            )

            # NFE 1
            d_1, denoised_1 = _derivative(
                model,
                x,
                sigma,
                s_in,
                extra_args,
            )

            # NFE 2
            sigma_half = (
                sigma
                + h * 0.5
            )

            x_2 = (
                x
                + d_1
                * (h * 0.5)
            )

            d_2, _ = _derivative(
                model,
                x_2,
                sigma_half,
                s_in,
                extra_args,
            )

            # NFE 3
            x_3 = (
                x
                + d_2
                * (h * 0.5)
            )

            d_3, _ = _derivative(
                model,
                x_3,
                sigma_half,
                s_in,
                extra_args,
            )

            # NFE 4
            x_4 = (
                x
                + d_3
                * h
            )

            d_4, _ = _derivative(
                model,
                x_4,
                sigma_down,
                s_in,
                extra_args,
            )

            # Classical RK4 integration.
            x_next_det = (
                x
                + (h / 6.0)
                * (
                    d_1
                    + 2.0 * d_2
                    + 2.0 * d_3
                    + d_4
                )
            )

            # ------------------------------------------------------
            # IMPORTANT NUMERICAL DETAIL
            #
            # Do NOT seed d_4 into the multistep history here.
            #
            # d_4 belongs to sigma_down.
            #
            # After this block we may inject ER-SDE noise with sigma_up,
            # producing a state whose effective sigma is sigma_next.
            #
            # Therefore:
            #
            #     derivative(d_4, sigma_down)
            #
            # is NOT the derivative of:
            #
            #     noisy_state(sigma_next)
            #
            # Seeding d_4 would lie to the multistep solver about where
            # that derivative lives.
            #
            # The next step performs one fresh NFE and begins again with
            # AB1. This is substantially safer than contaminating AB2/AB3.
            # ------------------------------------------------------

            history.clear()
            just_restarted = False

            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised_1,
                })

        # ==========================================================
        # PHASE 2
        #
        # VARIABLE-STEP 3RD-ORDER MULTISTEP CRUISING
        #
        # One NFE.
        #
        # The current derivative is evaluated at the actual current
        # sigma. The integration weights are then calculated from the
        # actual sigma locations in the history buffer.
        # ==========================================================

        else:

            d_i, denoised_i = _derivative(
                model,
                x,
                sigma,
                s_in,
                extra_args,
            )

            # Newest derivative first.
            history.insert(
                0,
                (
                    sigma_value,
                    d_i,
                ),
            )

            # Keep the latest three derivatives.
            if len(history) > 3:
                history.pop()

            d_integral = _multistep_predict(
                history=history,
                sigma=sigma_value,
                sigma_target=sigma_down_value,
            )

            x_next_det = (
                x
                + d_integral
            )

            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised_i,
                })

        # ==========================================================
        # PHASE 5
        #
        # ER-SDE MICRO-NOISE
        #
        # Add the stochastic component only after the deterministic
        # trajectory has reached sigma_down.
        #
        # The final state therefore has target variance sigma_next.
        # ==========================================================

        if sigma_next_value > 0.0 and _scalar_sigma(sigma_up) > 0.0:

            if noise_sampler is not None:
                eps_sde = noise_sampler(
                    sigma_down,
                    sigma_next,
                )
            else:
                eps_sde = torch.randn_like(x)

            x = (
                x_next_det
                + eps_sde
                * sigma_up
                * float(s_noise)
            )

        else:
            x = x_next_det

    return x
