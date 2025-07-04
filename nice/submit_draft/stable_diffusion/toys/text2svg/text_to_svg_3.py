import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pynvml
import random
import os

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import bitmap2svg

# --- Step 1: Helper Functions for Differentiable Vector Style Loss ---

def total_variation_loss(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Total Variation Loss for a batch of images.
    This loss encourages smoothness in the image by penalizing differences
    between adjacent pixels. It's a great way to encourage piecewise-constant regions.
    Args:
        img_tensor: A tensor of shape (B, C, H, W).
    Returns:
        A scalar tensor representing the TV loss.
    """
    # Difference vertically
    dy = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    # Difference horizontally
    dx = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    
    # Sum of absolute differences
    tv = torch.sum(dx) + torch.sum(dy)
    return tv

def palette_loss(img_tensor: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """
    Calculates the loss that encourages image pixels to be close to a fixed color palette.
    Args:
        img_tensor: Image tensor of shape (B, C, H, W). Assumed to be in [-1, 1] range.
        palette: Palette tensor of shape (N_colors, C). Assumed to be in [-1, 1] range.
    Returns:
        A scalar tensor representing the palette loss.
    """
    # Reshape image to a list of pixels
    # (B, C, H, W) -> (B*H*W, C)
    img_flat = img_tensor.permute(0, 2, 3, 1).reshape(-1, img_tensor.shape[1])
    
    # Calculate pairwise squared distances between each pixel and each palette color
    # `dists` will have shape (num_pixels, num_palette_colors)
    dists = torch.cdist(img_flat, palette, p=2).pow(2)
    
    # For each pixel, find the distance to the closest palette color
    min_dists, _ = torch.min(dists, dim=1)
    
    # The loss is the mean of these minimum distances
    return torch.mean(min_dists)

def calculate_vector_style_guidance(
    latents: torch.Tensor,
    vae: AutoencoderKL,
    t: torch.Tensor,
    scheduler: DDIMScheduler, # ADDED: scheduler is now needed here
    noise_pred: torch.Tensor, # ADDED: noise_pred is now needed here
    color_palette: torch.Tensor,
    tv_weight: float = 1.0,
    palette_weight: float = 1.0
) -> torch.Tensor:
    """
    Calculates the guidance gradient based on TV loss and Palette loss.
    """
    # 1. We need gradients on the latents
    latents = latents.detach().requires_grad_(True)
    
    # 2. This connects the grad-enabled `latents` to the final loss.
    pred_original_sample = scheduler.step(noise_pred, t, latents).pred_original_sample
    
    # 3. Use `vae.config.scaling_factor` as suggested by the warning.
    decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
    
    # 4. Calculate the style losses on the decoded image
    tv_loss = total_variation_loss(decoded_image_tensor)
    p_loss = palette_loss(decoded_image_tensor, color_palette)
    
    # 5. Combine the losses
    total_loss = (tv_loss * tv_weight) + (p_loss * palette_weight)
    
    # 6. Backpropagate the total loss to get gradients
    grad = torch.autograd.grad(total_loss, latents)[0]
    
    return grad


# --- Step 2: Main Generation Function with the Optimization Loop (Corrected) ---
# ... (imports and helper functions remain the same) ...

def generate_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5, # Text guidance scale
    vector_guidance_scale: float = 2.0, # Strength of the pixel-level vector style guidance
    tv_weight: float = 1.0,
    palette_weight: float = 1.0,
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    output_path: str = "output_guided_svg.svg",
    seed: int | None = None,
    # New parameter for two-stage generation
    enable_two_stage_generation: bool = False,
    first_stage_output_path: str = "first_stage_image.png"
):
    """
    Main function to generate an SVG from a text prompt using iterative style guidance.
    Optionally, enable a two-stage process: first generate a good raster, then guide.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=device).manual_seed(seed)

    # --- A. Load Models ---
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae_scale_factor = 8
    height = unet.config.sample_size * vae_scale_factor
    width = unet.config.sample_size * vae_scale_factor

    # --- B. Prepare Inputs ---
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (1, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
        generator=generator,
        device=device
    )
    latents = latents * scheduler.init_noise_sigma
    
    # Define a smaller, more contrasting palette
    color_palette = torch.tensor([
        [-1.0, -1.0, -1.0], # Black
        [ 1.0,  1.0,  1.0], # White
        [ 1.0, -1.0, -1.0], # Red
        [-1.0,  1.0, -1.0], # Green
        [-1.0, -1.0,  1.0], # Blue
        [ 1.0,  1.0, -1.0], # Yellow
        [ 0.0,  0.0,  0.0], # Mid Gray
    ], device=device)

    # --- TWO STAGE GENERATION ---
    if enable_two_stage_generation:
        # Temporarily disable vector guidance for the first stage
        original_vector_guidance_scale = vector_guidance_scale
        vector_guidance_scale = 0.0

        scheduler.set_timesteps(num_inference_steps)
        initial_latents = torch.randn(
            (1, unet.config.in_channels, height // vae_scale_factor, width // vae_scale_factor),
            generator=generator,
            device=device
        )
        initial_latents = initial_latents * scheduler.init_noise_sigma

        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Stage 1 Denoising")):
            latent_model_input = torch.cat([initial_latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            initial_latents = scheduler.step(noise_pred, t, initial_latents).prev_sample
        
        # Decode and save the first stage image
        initial_latents = 1 / vae.config.scaling_factor * initial_latents
        with torch.no_grad():
            initial_image_tensor = vae.decode(initial_latents).sample
        initial_image_tensor = (initial_image_tensor / 2 + 0.5).clamp(0, 1)
        initial_image_np = (initial_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        initial_pil_image = Image.fromarray(initial_image_np)
        initial_pil_image.save(first_stage_output_path)

        # Re-encode the initial image as latents for the second stage (Img2Img-like)
        # This re-encoding step can sometimes add noise, but ensures the latents are derived from a "good" image.
        with torch.no_grad():
            latents = vae.encode(initial_image_tensor * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
        
        # Restore vector guidance scale for the second stage
        vector_guidance_scale = original_vector_guidance_scale
        # Reduce inference steps for second stage as initial structure is already there
        num_inference_steps_stage2 = int(num_inference_steps * 0.7) # e.g., use 70% of steps
        scheduler.set_timesteps(num_inference_steps_stage2) # Set timesteps for the second stage
        # Adjust guidance start/end for the new step count
        guidance_start_step_stage2 = int(num_inference_steps_stage2 * 0.05) # Start early in stage 2
        guidance_end_step_stage2 = int(num_inference_steps_stage2 * 0.8) # End later
    else:
        num_inference_steps_stage2 = num_inference_steps
        guidance_start_step_stage2 = guidance_start_step
        guidance_end_step_stage2 = guidance_end_step
        scheduler.set_timesteps(num_inference_steps)


    # --- C. Denoising and Guidance Loop (Applies to single stage or second stage) ---
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Vector Guided Denoising")):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Apply vector guidance only in the specified range
        if guidance_start_step_stage2 <= i < guidance_end_step_stage2:
            grad = calculate_vector_style_guidance(
                latents, vae, t, scheduler, noise_pred, color_palette,
                tv_weight=tv_weight, palette_weight=palette_weight
            )
            
            # Apply guidance with a dynamically adjusted step size based on noise level
            # Using (1 - scheduler.alphas_cumprod[t])**0.5 is a common scaling in SDS
            # A smaller step size for guidance is usually safer, so vector_guidance_scale acts as a learning rate.
            grad_scale = (1 - scheduler.alphas_cumprod[t])**0.5
            latents = latents - grad * grad_scale * vector_guidance_scale
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # --- D. Final Image and SVG Generation ---
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    
    raster_output_path = os.path.splitext(output_path)[0] + ".png"
    final_image.save(raster_output_path)
    final_svg_params = {
        'num_colors': 7, # Match palette size or slightly less for more generalization
        'simplification_epsilon_factor': 0.01, # Further simplification
        'min_contour_area': 15.0, # Increase to filter more noise
        'max_features_to_render': 0
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image, **final_svg_params)
    
    with open(output_path, "w") as f:
        f.write(final_svg_string)

    return final_image, final_svg_string

# --- Example Usage ---
if __name__ == '__main__':
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU found. Free memory: {meminfo.free / 1024**2:.2f} MB")
        if meminfo.free < 5 * 1024**3: # Check for 5GB VRAM
             print("Warning: Low VRAM, generation might fail.")
        DEVICE = "cuda"
    except (pynvml.NVMLError, FileNotFoundError):
        print("No NVIDIA GPU found or NVML library not installed. Using CPU. This will be very slow.")
        DEVICE = "cpu"

    # --- Parameters ---
    PROMPT = "a simple minimalistic logo of a lion, vector art, flat colors, clear lines, no gradients, abstract, bold shapes, cartoon"
    NEGATIVE_PROMPT = "blurry, noisy, distorted, messy, complex, detailed, photographic, realistic, too much texture, shadows, gradients, fine details"
    
    NUM_STEPS = 50 # Total steps for generation
    
    # Adjust these weights for stronger pixel-level vector aesthetic enforcement
    VECTOR_GUIDANCE_SCALE = 8.0 # Increased guidance scale for more impact
    TV_WEIGHT = 3.0 # Significantly increase TV weight for more smoothness
    PALETTE_WEIGHT = 5.0 # Significantly increase palette weight for quantized colors

    GUIDANCE_START = int(NUM_STEPS * 0.05) # Start early
    GUIDANCE_END = int(NUM_STEPS * 0.7) # End a bit earlier to let SD stabilize

    # Set to True to enable the two-stage generation process
    ENABLE_TWO_STAGE = True 

    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        device=DEVICE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=12, # A common text guidance scale
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        tv_weight=TV_WEIGHT,
        palette_weight=PALETTE_WEIGHT,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        output_path="lion_logo_guided_refined.svg",
        seed=42, # Use a fixed seed for reproducibility
        enable_two_stage_generation=ENABLE_TWO_STAGE,
        first_stage_output_path="lion_logo_first_stage.png"
    )
