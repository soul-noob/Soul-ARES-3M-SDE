# ======================================================================
# FORGE / AUTOMATIC1111 INJECTION SCRIPT
# Hooks the Soul_ARES mathematical engine directly into the WebUI dropdown
# ======================================================================

import sys
import os
import k_diffusion.sampling as k_sampling
from modules import sd_samplers, sd_samplers_kdiffusion, sd_samplers_common

# 1. Add the main repository folder to the path so A1111/Forge can find your math engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampler_core import sample_restart_er_3ma

SAMPLER_NAME = "soul_ares_3m_sde"
UI_NAME = "Soul ARES 3M SDE"

# 2. Attach our sampler to the internal k_diffusion backend
if not hasattr(k_sampling, f'sample_{SAMPLER_NAME}'):
    setattr(k_sampling, f'sample_{SAMPLER_NAME}', sample_restart_er_3ma)

# 3. Create the WebUI-specific Sampler Data Object
new_sampler = sd_samplers_common.SamplerData(
    UI_NAME, 
    lambda model: sd_samplers_kdiffusion.KDiffusionSampler(f'sample_{SAMPLER_NAME}', model), 
    [SAMPLER_NAME], 
    {}
)

# 4. Inject it into the UI Dropdown (and prevent duplicates on UI-Reload)
if not any(s.name == UI_NAME for s in sd_samplers.all_samplers):
    sd_samplers.all_samplers.append(new_sampler)
    # Refresh the internal sampler list so it appears immediately
    sd_samplers.set_samplers()
    print(f"  [+] Loaded Soul_Noob's Hybrid Sampler into WebUI: {UI_NAME}")
