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
def generate_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    vector_guidance_scale: float = 2.0,
    tv_weight: float = 1.0,
    palette_weight: float = 1.0,
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    output_path: str = "output_guided_svg.svg",
    seed: int | None = None
):
    """
    Main function to generate an SVG from a text prompt using iterative style guidance.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Using seed: {seed}")

    # --- A. Load Models ---
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # --- B. Prepare Inputs ---
    print("Preparing inputs...")
    
    vae_scale_factor = 8
    height = unet.config.sample_size * vae_scale_factor
    width = unet.config.sample_size * vae_scale_factor
    
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
    
    # --- C. Denoising and Guidance Loop ---
    print("Starting denoising and guidance loop...")
    scheduler.set_timesteps(num_inference_steps)
    
    simple_palette_rgb = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5], [0.8, 0.5, 0.2], [0.5, 0.2, 0.8], [0.2, 0.8, 0.5]
    ], device=device)
    color_palette = simple_palette_rgb * 2 - 1

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if guidance_start_step <= i < guidance_end_step:
            # FIX: Call the guidance function with the new required parameters
            grad = calculate_vector_style_guidance(
                latents, vae, t, scheduler, noise_pred, color_palette,
                tv_weight=tv_weight, palette_weight=palette_weight
            )
            
            grad_scale = (1 - scheduler.alphas_cumprod[t])**0.5
            latents = latents - grad * grad_scale * vector_guidance_scale
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # --- D. Final Image and SVG Generation ---
    print("Generating final image and SVG...")
    
    # FIX: Use `vae.config.scaling_factor` as suggested by the warning.
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    
    raster_output_path = os.path.splitext(output_path)[0] + ".png"
    final_image.save(raster_output_path)
    print(f"Saved final raster image to: {raster_output_path}")

    print("Performing final vectorization...")
    final_svg_params = {
        'num_colors': 16,
        'simplification_epsilon_factor': 0.005,
        'min_contour_area': 5.0,
        'max_features_to_render': 0
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image, **final_svg_params)
    
    with open(output_path, "w") as f:
        f.write(final_svg_string)
    print(f"Saved final SVG to: {output_path}")

    return final_image, final_svg_string

# --- Example Usage ---
if __name__ == '__main__':
    # Check for GPU
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
    PROMPT = "a lighthouse overlooking the ocean"
    NEGATIVE_PROMPT = ""
    
    # Lower steps for faster testing, increase to 50 for quality
    NUM_STEPS = 27
    
    # Strength of the vector guidance. Higher values force a more "vector-like" style.
    # This value might need different tuning compared to the SDS approach.
    VECTOR_GUIDANCE_SCALE = 2.5
    
    # Relative weights of the two style loss components.
    TV_WEIGHT = 1.0 # Encourages smooth areas
    PALETTE_WEIGHT = 1.0 # Encourages use of the color palette

    # When to apply the guidance.
    GUIDANCE_START = int(NUM_STEPS * 0.1)
    GUIDANCE_END = int(NUM_STEPS * 0.8)

    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        device=DEVICE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=20,
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        tv_weight=TV_WEIGHT,
        palette_weight=PALETTE_WEIGHT,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        output_path="lighthouse_guided.svg",
        seed=42 # Use a fixed seed for reproducibility
    )