# ======================================================================
# SOUL ARES 3M SDE
# ComfyUI sampler registration
# ======================================================================

import comfy.samplers
import comfy.k_diffusion.sampling as k_sampling

from .sampler_core import sample_restart_er_3ma


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SAMPLER_NAME = "soul_ares_3m_sde"
SAMPLER_FUNCTION_NAME = f"sample_{SAMPLER_NAME}"


# ----------------------------------------------------------------------
# 1. Register the sampler implementation with k-diffusion
# ----------------------------------------------------------------------
#
# ComfyUI resolves sampler implementations using:
#
#     sample_<sampler_name>
#
# Therefore:
#
#     soul_ares_3m_sde
#
# must resolve to:
#
#     sample_soul_ares_3m_sde
#
# ----------------------------------------------------------------------

setattr(
    k_sampling,
    SAMPLER_FUNCTION_NAME,
    sample_restart_er_3ma,
)


# ----------------------------------------------------------------------
# 2. Register the sampler in ComfyUI's sampler lists
# ----------------------------------------------------------------------
#
# IMPORTANT:
#
# ComfyUI has both:
#
#   KSAMPLER_NAMES
#   SAMPLER_NAMES
#
# SAMPLER_NAMES is constructed separately from KSAMPLER_NAMES, so adding
# a name only to KSAMPLER_NAMES does NOT necessarily make it appear in
# the KSampler node's dropdown.
#
# We therefore explicitly register it in both.
# ----------------------------------------------------------------------

if SAMPLER_NAME not in comfy.samplers.KSAMPLER_NAMES:
    comfy.samplers.KSAMPLER_NAMES.append(SAMPLER_NAME)

if SAMPLER_NAME not in comfy.samplers.SAMPLER_NAMES:
    comfy.samplers.SAMPLER_NAMES.append(SAMPLER_NAME)


# ----------------------------------------------------------------------
# 3. Node mappings
# ----------------------------------------------------------------------
#
# We are extending ComfyUI's native KSampler rather than creating a
# separate custom node.
# ----------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ----------------------------------------------------------------------
# 4. Startup message
# ----------------------------------------------------------------------

print("=" * 70)
print("  Soul-ARES 3M SDE loaded")
print(f"  Sampler: {SAMPLER_NAME}")
print(f"  Function: {SAMPLER_FUNCTION_NAME}")
print("  Engine: Restart + ER-SDE + DPM++ 3M + aRK4")
print("=" * 70)


# ----------------------------------------------------------------------
# 5. Optional package metadata
# ----------------------------------------------------------------------

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
