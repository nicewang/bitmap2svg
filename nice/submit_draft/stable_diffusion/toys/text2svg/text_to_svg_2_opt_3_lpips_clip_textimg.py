# ==============================================================================
# Text-to-SVG Generation with Stable Diffusion, Hybrid Loss, and Multi-GPU
#
# Features:
# - Stable Diffusion for base image generation.
# - Vector guidance using a hybrid loss (Reconstruction + Semantic).
# - Multi-GPU support for parallel computation.
# - Various memory optimization techniques.
# - VERSION: Added logic to initialize latents from text embeddings.
# ==============================================================================

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import re
import os
import random
import gc

import kagglehub

# 1. Import necessary libraries
import accelerate
import lpips # Import the LPIPS library

# NEW: Import CLIP-related components from transformers
from transformers import CLIPModel, CLIPProcessor

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Differentiable SVG Renderer
import pydiffvg

import bitmap2svg

import logging

print("Downloading Stable Diffusion model from Kaggle Hub...")
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
print("Model download complete.")

# The parse_svg_and_render function remains the same and is assumed to be defined here.
def parse_svg_and_render(svg_string: str, width: int, height: int, device: str) -> torch.Tensor:
    polygons = re.findall(r'<polygon points="([^"]+)" fill="([^"]+)"/>', svg_string)
    shapes, shape_groups = [], []
    for points_str, fill_str in polygons:
        try:
            points_data = [float(p) for p in points_str.replace(',', ' ').split()]
            if not points_data or len(points_data) % 2 != 0: continue
            points = torch.tensor(points_data, dtype=torch.float32, device=device).view(-1, 2)
            hex_color = fill_str.lstrip('#')
            if len(hex_color) == 3: r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
            else: r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color = torch.tensor([r/255.0, g/255.0, b/255.0, 1.0], device=device)
            path = pydiffvg.Polygon(points=points, is_closed=True)
            shapes.append(path)
            shape_groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1], device=device), fill_color=color))
        except (ValueError, IndexError):
            continue

    bg_match = re.search(r'<rect .* fill="([^"]+)"/>', svg_string)
    bg_color_tensor = torch.tensor([0.0, 0.0, 0.0], device=device)
    if bg_match:
        hex_color = bg_match.group(1).lstrip('#')
        if len(hex_color) == 3: r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
        else: r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_color_tensor = torch.tensor([r/255.0, g/255.0, b/255.0], device=device)

    if not shapes:
        return bg_color_tensor.view(1, 3, 1, 1).expand(1, 3, height, width)

    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(width, height, 2, 2, 0, None, *scene_args)
    img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * bg_color_tensor
    img = img.unsqueeze(0).permute(0, 3, 1, 2)

    return img


# --- NEW HELPER FUNCTION ---
def create_latents_from_embedding(embedding: torch.Tensor, target_shape: tuple, generator: torch.Generator) -> torch.Tensor:
    """
    Creates a structured initial latent tensor from a text embedding.
    """
    # embedding shape is (1, embedding_dim), e.g., (1, 1024)
    # target_shape is (batch_size, channels, height, width), e.g., (1, 4, 64, 64)
    batch_size, channels, height, width = target_shape
    emb_dim = embedding.shape[1]

    # Ensure the embedding is on the correct device
    device = embedding.device

    # Reshape and interpolate the embedding to create a structured latent
    # We want to create 'channels' number of images from the embedding
    chunk_size = emb_dim // channels
    
    if chunk_size == 0:
        # Fallback for small embeddings
        return torch.randn(target_shape, generator=generator, device=device)
        
    # Calculate a side length for a square we can form from each chunk
    side_len = int(chunk_size**0.5)
    
    # Add a check for side_len being zero to prevent errors
    if side_len == 0:
        return torch.randn(target_shape, generator=generator, device=device)
        
    structured_channels = []
    for i in range(channels):
        chunk = embedding[0, i*chunk_size : (i+1)*chunk_size]
        # Take a slice that can form a square
        square_chunk = chunk[:side_len*side_len]
        # Reshape to a small square image
        small_image = square_chunk.view(1, 1, side_len, side_len)
        # Interpolate to the target latent size
        interpolated_channel = F.interpolate(small_image, size=(height, width), mode='bilinear', align_corners=False)
        structured_channels.append(interpolated_channel)

    structured_latent = torch.cat(structured_channels, dim=1)
    
    # Normalize the structured latent to have a similar distribution to random noise
    if structured_latent.std() > 0:
        structured_latent = (structured_latent - structured_latent.mean()) / structured_latent.std()
    
    return structured_latent.to(device)


def generate_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    description: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda:0",
    # --- Strength parameter for blending structured and random noise ---
    strength: float = 0.8, # 1.0 is pure noise, 0.0 is pure structure
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    vector_guidance_scale: float = 1.5,
    lpips_mse_lambda: float = 0.1,
    clip_guidance_scale: float = 0.5, 
    clip_model_id: str = "openai/clip-vit-large-patch14",
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    guidance_resolution: int = 256,
    guidance_interval: int = 2,
    seed: int | None = None,
    enable_attention_slicing: bool = True,
    use_half_precision: bool = True,
    batch_size: int = 1,
    enable_sequential_cpu_offload: bool = True,
    low_vram_shift_to_cpu: bool = True
):

    model_id = stable_diffusion_path
    
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"Strength must be between 0.0 and 1.0, but got {strength}")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=device).manual_seed(seed)
    
    gc.collect()
    torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # This logic now uses a generic `guidance_device` which defaults to cuda:1
    # or the main device if cuda:1 is not available.
    guidance_device = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device
    
    print(f"Initializing LPIPS model on {guidance_device}...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(guidance_device).eval()

    print(f"Initializing CLIP model and processor on {guidance_device}...")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(guidance_device).eval()
    
    dtype = torch.float16 if use_half_precision else torch.float32
    loading_kwargs = {"torch_dtype": dtype, "use_safetensors": True, "low_cpu_mem_usage": True}
    if use_half_precision: loading_kwargs["variant"] = "fp16"

    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **loading_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **loading_kwargs)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **loading_kwargs)
    except Exception as e:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    if enable_attention_slicing:
        # Use a robust check for different diffusers versions
        if hasattr(unet, 'enable_sliced_attention'):
            unet.enable_sliced_attention()
        elif hasattr(unet, 'enable_attention_slicing'):
            unet.enable_attention_slicing()

    if enable_sequential_cpu_offload:
        try:
            unet = accelerate.cpu_offload(unet, execution_device=device)
            vae = accelerate.cpu_offload(vae, execution_device=device)
            text_encoder = accelerate.cpu_offload(text_encoder, execution_device=device)
        except Exception as e:
            text_encoder, unet, vae = text_encoder.to("cpu"), unet.to("cpu"), vae.to("cpu")
            enable_sequential_cpu_offload = False
            low_vram_shift_to_cpu = True
    else:
        text_encoder, unet, vae = text_encoder.to("cpu"), unet.to("cpu"), vae.to("cpu")

    # --- Input Preparation ---
    height = 512
    width = 512

    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to(device)
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        prompt_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        if description != "":
            des_input = tokenizer([description], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            input_embeddings = text_encoder(des_input.input_ids.to(device))[0]
        else:
            input_embeddings = prompt_embeddings
        
        uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to("cpu")
        
        text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings]).to(device=device, dtype=dtype)
        
        clip_text_input = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(guidance_device)
        text_features = clip_model.get_text_features(**clip_text_input)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    latent_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)
    
    structured_latents = create_latents_from_embedding(input_embeddings.mean(dim=1), latent_shape, generator)
    
    # Create pure random noise
    random_latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    
    # Blend the structured and random latents based on the strength parameter
    latents = (1.0 - strength) * structured_latents + strength * random_latents
    

    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    svg_params_guidance = {
        'num_colors': None,
        'simplification_epsilon_factor': 0.02,
        'min_contour_area': (guidance_resolution/512)**2 * 30.0,
        'max_features_to_render': 64
    }

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        if i % 5 == 0: gc.collect(); torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: unet = unet.to(device)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: unet = unet.to("cpu")

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if guidance_start_step <= i < guidance_end_step and i % guidance_interval == 0:
            with torch.no_grad():
                pred_original_sample = scheduler.step(noise_pred_cfg, t, latents).pred_original_sample
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to(device)
                decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to("cpu")

                target_image_for_loss = decoded_image_tensor
                if guidance_resolution < height:
                    target_image_for_loss = F.interpolate(decoded_image_tensor.float(), size=(guidance_resolution, guidance_resolution), mode='bilinear', align_corners=False).to(dtype)

                img_to_vectorize_scaled = (target_image_for_loss / 2 + 0.5).clamp(0, 1)
                image_np = (img_to_vectorize_scaled.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                svg_string = bitmap2svg.bitmap_to_svg(pil_image, **svg_params_guidance)
                rendered_svg_tensor = parse_svg_and_render(svg_string, pil_image.width, pil_image.height, device)
                rendered_svg_tensor_scaled = rendered_svg_tensor * 2.0 - 1.0

                # Switch to guidance device for loss calculation
                target_image_gpu = target_image_for_loss.to(guidance_device)
                rendered_svg_gpu = rendered_svg_tensor_scaled.to(guidance_device)

                loss_lpips_val = loss_fn_lpips(target_image_gpu.float(), rendered_svg_gpu.float()).mean()
                loss_mse_val = F.mse_loss(target_image_gpu, rendered_svg_gpu)
                reconstruction_loss = loss_lpips_val + lpips_mse_lambda * loss_mse_val

                clip_image_input = clip_processor(images=rendered_svg_tensor, return_tensors="pt").to(guidance_device)
                
                image_features = clip_model.get_image_features(**clip_image_input)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                clip_loss = 1 - (text_features @ image_features.T).squeeze()
                total_loss = reconstruction_loss + clip_guidance_scale * clip_loss

                grad = noise_pred_text - noise_pred_uncond
                noise_pred_cfg = noise_pred_cfg + (grad * total_loss.item() * vector_guidance_scale)

            gc.collect(); torch.cuda.empty_cache()

        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to(device)
        latents = 1 / vae.config.scaling_factor * latents
        image_tensor = vae.decode(latents).sample
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to("cpu")

    image_tensor = (image_tensor.cpu() / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)

    final_svg_params = {
        'num_colors': None, 'simplification_epsilon_factor': 0.002,
        'min_contour_area': 0.5, 'max_features_to_render': 0
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image, **final_svg_params)
    
    del vae, text_encoder, unet, tokenizer, scheduler, loss_fn_lpips, clip_model, clip_processor
    gc.collect()
    torch.cuda.empty_cache()

    return final_image, final_svg_string


# prompt = "A cute cat wearing a wizard hat, minimalist logo, vector art"
# negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render"

# # prompt = "gray wool coat with a faux fur collar, vector art"
# prompt = "simple vector illustration of gray wool coat with a faux fur collar"
# negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render"

description = "a maroon dodecahedron interwoven with teal threads"
# prompt = (
#     f"Simple vector illustration of {description} "
#     "with flat color blocks, beautiful, minimal details, solid colors only"
# )
# negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"

prompt = f"simple vector illustration of {description}"
negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render, lines, framing, hatching, background, textures, patterns, details, outlines"

seed = 42
device = "cuda:0"

img, svg = generate_svg_with_guidance(
    prompt=prompt,
    negative_prompt=negative_prompt,
    description=description,
    device=device,
    # --- Strength parameter for blending structured and random noise ---
    strength=0.8, # 1.0 is pure noise, 0.0 is pure structure
    num_inference_steps=25,
    guidance_scale=20,
    vector_guidance_scale=3.5,
    lpips_mse_lambda=0.1,
    clip_guidance_scale=0.8, 
    guidance_start_step=0,
    guidance_end_step=24,
    guidance_resolution=256,
    guidance_interval=1,
    seed=42,
    use_half_precision=True,
    enable_sequential_cpu_offload=True,
    low_vram_shift_to_cpu=False
)
