# Soul ARES 3M SDE

A custom hybrid sampler for ComfyUI, Forge, and AUTOMATIC1111.

Soul ARES (Adaptive aRK4 + Restart + ER-SDE + 3M) combines multiple solver methods to improve texture details, anatomical consistency, and dynamic self-correction, all while maintaining a strict NFE (Neural Function Evaluation) budget of under 40 steps.

## How It Works

The sampler actively manages the NFE budget by swapping solvers dynamically during generation:

* **Midpoint Bootstrap:** Replaces the standard Heun Step 0 with a Midpoint evaluation. By evaluating the exact center of the curve, it establishes a more accurate initial composition foundation.
* **DPM++ 3M Engine:** Runs for the majority of the generation at 1 NFE/step, utilizing a 3rd-order history buffer to pull out high-frequency micro-details.
* **Dynamic Restart with Targeted aRK4:** Injects macro-noise jumps dynamically based on your step count to allow the model to fix anatomical errors. During the recovery step, it temporarily switches to a 4-stage Runge-Kutta (aRK4) solver to stabilize the time-travel gap without breaking the 3M history buffer.
* **Variance-Locked ER-SDE:** Injects micro-noise matching the physical diffusion coefficients. It includes a variance clamp to cap noise injection, preventing image burn-in at high CFG scales or during aggressive Restart jumps.

## Installation

### For ComfyUI
1. Go to your `ComfyUI/custom_nodes/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Soul-ARES-3M-SDE.git
   ```
3. Restart ComfyUI. 
4. The sampler will appear natively in your standard `KSampler` node dropdown as **`soul_ares_3m_sde`**.

### For Forge & AUTOMATIC1111 (WebUI)
1. Go to your `webui/extensions/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Soul-ARES-3M-SDE.git
   ```
3. Restart your WebUI.
4. The sampler will appear natively in your Sampling Method dropdown as **`Soul ARES 3M SDE`**.

## Recommended Settings

This sampler uses high-order math and dynamic sub-stepping. It is optimized for lower step counts. 

* **Steps:** `20 to 35` (Scaling past 40 steps is not recommended and provides diminishing returns).
* **Scheduler (FLUX / SD3.5 / Anima):** `simple` (Required for Flow-Matching models).
* **Scheduler (SDXL / SD1.5):** `sgm_uniform` (Maximizes the stability of the 3M engine).
* **CFG Scale:** Standard ranges for your model. The variance lock will prevent deep-frying at reasonable values.

## License

Created by **Soul_Noob**.
Released under the **Apache License 2.0**.
