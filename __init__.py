# ======================================================================
# COMFYUI INJECTION NODE
# Hooks the Soul_ARES mathematical engine directly into the UI dropdowns
# ======================================================================

import comfy.samplers
import comfy.k_diffusion.sampling as k_sampling
from .sampler_core import sample_restart_er_3ma

# Your personalized sampler name as it will appear in the UI dropdown
SAMPLER_NAME = "soul_ares_3m_sde"

# 1. Attach our sampler to ComfyUI's internal k_diffusion backend.
# ComfyUI automatically looks for a function named "sample_" + SAMPLER_NAME.
# So we dynamically bind your math engine to that exact name.
if not hasattr(k_sampling, f'sample_{SAMPLER_NAME}'):
    setattr(k_sampling, f'sample_{SAMPLER_NAME}', sample_restart_er_3ma)

# 2. Add it to the standard KSampler Dropdown list safely
if SAMPLER_NAME not in comfy.samplers.KSAMPLER_NAMES:
    comfy.samplers.KSAMPLER_NAMES.append(SAMPLER_NAME)

# 3. Required extension mappings for ComfyUI. 
# Because we are upgrading the native KSampler directly, we don't need 
# to clutter the user's canvas with weird custom nodes. We leave these empty.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 4. Boot-up message so users know it loaded successfully
print(f"==================================================")
print(f"  [+] Loaded Soul_Noob's Hybrid Sampler: {SAMPLER_NAME} ")
print(f"      Engine: ARES (aRK4 + Restart + ER-SDE + 3M) ")
print(f"==================================================")
