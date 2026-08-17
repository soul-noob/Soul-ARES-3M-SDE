import torch
from tqdm.auto import trange

@torch.no_grad()
def sample_restart_er_3ma(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, gamma=0.15, s_noise=1.0, noise_sampler=None):
    """
    Ultimate Hybrid Sampler: Restart + ER-SDE + DPM++ 3M + aRK4 + Midpoint Bootstrap
    Mathematically tuned for <= 40 NFEs with RES4LYF-style variance locking.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # ---------------------------------------------------------
    # 1. DYNAMIC RESTART INJECTION (Targeting < 40 total NFEs)
    # ---------------------------------------------------------
    steps = len(sigmas) - 1
    new_sigmas = sigmas.tolist()
    
    # Check if the schedule is purely forward-moving (no restarts injected yet)
    is_decreasing = all(new_sigmas[i] >= new_sigmas[i+1] for i in range(len(new_sigmas)-1))
    
    if is_decreasing:
        if 15 <= steps <= 25:
            # 1 Restart at 50%
            k = steps // 2
            if k >= 3:
                new_sigmas = new_sigmas[:k+1] + new_sigmas[k-2:k+1] + new_sigmas[k+1:]
        elif 26 <= steps <= 40:
            # 2 Restarts at 33% and 66%
            k1 = steps // 3
            k2 = (steps * 2) // 3
            if k1 >= 3 and k2 >= 3:
                new_sigmas = new_sigmas[:k2+1] + new_sigmas[k2-2:k2+1] + new_sigmas[k2+1:]
                new_sigmas = new_sigmas[:k1+1] + new_sigmas[k1-2:k1+1] + new_sigmas[k1+1:]
                
    sigmas = torch.tensor(new_sigmas, device=sigmas.device, dtype=sigmas.dtype)
    
    # ---------------------------------------------------------
    # HELPER: Neural Function Evaluation (Derivative)
    # ---------------------------------------------------------
    def get_derivative(x_temp, sig_temp):
        denoised = model(x_temp, sig_temp * s_in, **extra_args)
        return (x_temp - denoised) / sig_temp

    # ---------------------------------------------------------
    # 2. THE DENOISING LOOP
    # ---------------------------------------------------------
    history = []
    just_restarted = False
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # --- PHASE A: MACRO-NOISE JUMP (RESTART TIME TRAVEL) ---
        if sigma_next > sigma:
            delta_sigma_sq = sigma_next**2 - sigma**2
            
            if noise_sampler is not None:
                # Pass inverted sigmas to prevent assertion errors in custom schedulers
                epsilon = noise_sampler(sigma_next, sigma)
            else:
                epsilon = torch.randn_like(x)
                
            x = x + epsilon * (delta_sigma_sq ** 0.5)
            
            # Clear the history buffer so DPM++ 3M doesn't explode
            history.clear()
            just_restarted = True
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x})
            continue  # Skip evaluation, loop to the next step
            
        # --- PHASE B: ER-SDE TARGETS WITH RES4LYF VARIANCE LOCK ---
        if sigma_next > 0:
            sigma_ratio = sigma_next / sigma
            ratio_sq = torch.clamp(1.0 - sigma_ratio**2, min=0.0)
            
            # Mathematical Target
            er_target = eta * sigma_next * (ratio_sq ** 0.5)
            # RES4LYF Clamp Threshold (prevents image burning)
            gamma_target = gamma * sigma_next 
            
            # Apply the floor lock
            sigma_up = torch.min(er_target, gamma_target)
            sigma_down = torch.clamp(sigma_next**2 - sigma_up**2, min=0.0) ** 0.5
        else:
            sigma_up = torch.zeros_like(sigma)
            sigma_down = torch.zeros_like(sigma)
            
        # --- PHASE C: THE SOLVERS ---
        
        # Condition 1: Step 0 Midpoint Bootstrap (2 NFE)
        if len(history) == 0 and i == 0:
            sigma_mid = (sigma + sigma_down) / 2.0
            d_0 = get_derivative(x, sigma)
            
            # Half step
            x_mid = x + d_0 * (sigma_mid - sigma)
            # Midpoint evaluation
            d_mid = get_derivative(x_mid, sigma_mid)
            
            # Full deterministic advance
            x_next_det = x + d_mid * (sigma_down - sigma)
            history.append(d_mid)
            
            if callback is not None:
                # We report the initial evaluation for UI consistency
                callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x - d_0 * sigma})
                
        # Condition 2: Targeted aRK4 Recovery (4 NFE, Triggers ONLY after a Restart)
        elif just_restarted:
            h = sigma_down - sigma
            
            d_1 = get_derivative(x, sigma)
            d_2 = get_derivative(x + d_1 * (h / 2.0), sigma + (h / 2.0))
            d_3 = get_derivative(x + d_2 * (h / 2.0), sigma + (h / 2.0))
            d_4 = get_derivative(x + d_3 * h, sigma_down)
            
            x_next_det = x + (h / 6.0) * (d_1 + 2 * d_2 + 2 * d_3 + d_4)
            
            # Seed the empty history buffer so DPM++ 3M can seamlessly take over
            history.append(d_4)
            just_restarted = False
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x - d_1 * sigma})
                
        # Condition 3: DPM++ 3M Engine (1 NFE, Core Cruising Speed)
        else:
            d_i = get_derivative(x, sigma)
            history.append(d_i)
            
            if len(history) > 3:
                history.pop(0)
                
            if len(history) == 1:
                d_pred = history[-1]
            elif len(history) == 2:
                d_pred = (3/2) * history[-1] - (1/2) * history[-2]
            else:
                d_pred = (23/12) * history[-1] - (16/12) * history[-2] + (5/12) * history[-3]
                
            x_next_det = x + d_pred * (sigma_down - sigma)
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x - d_i * sigma})

        # --- PHASE D: ER-SDE MICRO-NOISE INJECTION ---
        if sigma_next > 0:
            if noise_sampler is not None:
                eps_sde = noise_sampler(sigma, sigma_next)
            else:
                eps_sde = torch.randn_like(x)
                
            x = x_next_det + eps_sde * sigma_up
        else:
            x = x_next_det

    return x
