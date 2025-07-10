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

stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

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


def generate_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda:0",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    vector_guidance_scale: float = 1.5,
    # --- New Parameters for Hybrid Loss ---
    lpips_mse_lambda: float = 0.1,
    clip_guidance_scale: float = 0.5, # Weight for the new CLIP semantic loss
    clip_model_id: str = "openai/clip-vit-large-patch14", # The CLIP model to use for guidance
    # ---
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    guidance_resolution: int = 256,
    guidance_interval: int = 2,
    seed: int | None = None,
    # Memory optimization flags are preserved
    enable_attention_slicing: bool = True,
    use_half_precision: bool = True,
    batch_size: int = 1,
    enable_sequential_cpu_offload: bool = True,
    low_vram_shift_to_cpu: bool = True
):

    model_id = stable_diffusion_path

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=device).manual_seed(seed)

    gc.collect()
    torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # --- Initialize LPIPS model on cuda:1 to save VRAM ---
    print("Initializing LPIPS model on cuda:1...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to("cuda:1").eval()

    # --- NEW: Initialize CLIP model and processor on cuda:1 ---
    print("Initializing CLIP model and processor on cuda:1...")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda:1").eval()

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
        try:
            if hasattr(unet, 'enable_attention_slicing'): unet.enable_attention_slicing(1)
            elif hasattr(unet, 'set_attention_slice'): unet.set_attention_slice(1)
        except Exception as e: logging.error(f"Could not enable attention slicing: {e}")

    if enable_sequential_cpu_offload:
        try:
            text_encoder, unet, vae = text_encoder.to("cpu"), unet.to("cpu"), vae.to("cpu")
            unet = accelerate.cpu_offload(unet, execution_device=device)
            vae = accelerate.cpu_offload(vae, execution_device=device)
            text_encoder = accelerate.cpu_offload(text_encoder, execution_device=device)
        except Exception as e:
            text_encoder, unet, vae = text_encoder.to("cpu"), unet.to("cpu"), vae.to("cpu")
            enable_sequential_cpu_offload = False
            low_vram_shift_to_cpu = True
    else:
        text_encoder, unet, vae = text_encoder.to("cpu"), unet.to("cpu"), vae.to("cpu")

    try:
        if hasattr(unet, 'enable_xformers_memory_efficient_attention'):
            unet.enable_xformers_memory_efficient_attention()
    except (ImportError, AttributeError, Exception) as e:
        logging.error(f"Could not enable xFormers: {e}")

    try:
        if hasattr(unet, 'enable_gradient_checkpointing'):
            unet.enable_gradient_checkpointing()
    except Exception as e:
        logging.error(f"Could not enable gradient checkpointing: {e}")

    # --- Input Preparation ---
    gc.collect()
    torch.cuda.empty_cache()
    height = 512
    width = 512

    with torch.no_grad():
        # --- SD Text Embeddings ---
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to(device)
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to("cpu")

        uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to(device)
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: text_encoder = text_encoder.to("cpu")

        # --- NEW: Pre-calculate CLIP Text Features for guidance ---
        clip_model.to(device) # Move to GPU for this calculation
        clip_text_input = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**clip_text_input)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) # Normalize features
        clip_model.to("cpu") # Move back to CPU immediately
        # ---

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(device=device, dtype=dtype)
    del uncond_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    latent_height = int(height // 8)
    latent_width = int(width // 8)
    latents = torch.randn((batch_size, unet.config.in_channels, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps)

    svg_params_guidance = {
        'num_colors': None,
        'simplification_epsilon_factor': 0.02,
        'min_contour_area': (guidance_resolution/512)**2 * 30.0,
        'max_features_to_render': 64
    }

    # --- Denoising Loop with Hybrid Perceptual + Semantic Loss ---
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        if i % 5 == 0: gc.collect(); torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: unet = unet.to(device)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: unet = unet.to("cpu")

        del latent_model_input
        gc.collect(); torch.cuda.empty_cache()

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

                # --- HYBRID LOSS CALCULATION ---
                # 1. Reconstruction Loss (LPIPS + MSE)
                loss_fn_lpips.to(device)
                loss_lpips_val = loss_fn_lpips(target_image_for_loss.float(), rendered_svg_tensor_scaled.float()).mean()
                loss_fn_lpips.to("cpu")
                loss_mse_val = F.mse_loss(target_image_for_loss, rendered_svg_tensor_scaled)
                reconstruction_loss = loss_lpips_val + lpips_mse_lambda * loss_mse_val

                # 2. NEW: Semantic Loss (CLIP)
                clip_model.to(device) # Move CLIP model to GPU for calculation
                # Process the rendered SVG image for CLIP. Note the image range for CLIP is [0, 1]
                clip_image_input = clip_processor(images=rendered_svg_tensor, return_tensors="pt").to(device)
                image_features = clip_model.get_image_features(**clip_image_input)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True) # Normalize
                clip_model.to("cpu") # Immediately offload back to CPU
                
                # Calculate loss as 1 - cosine_similarity
                clip_loss = 1 - (text_features @ image_features.T).squeeze()

                # 3. Combine losses
                total_loss = reconstruction_loss + clip_guidance_scale * clip_loss

                # Apply guidance based on the total combined loss
                grad = noise_pred_text - noise_pred_uncond
                noise_pred_cfg = noise_pred_cfg + (grad * total_loss.item() * vector_guidance_scale)

                del pred_original_sample, decoded_image_tensor, rendered_svg_tensor, rendered_svg_tensor_scaled, target_image_for_loss
                del img_to_vectorize_scaled, pil_image, svg_string, image_features, clip_image_input

            gc.collect(); torch.cuda.empty_cache()

        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample

    # --- Final Generation (Preserved) ---
    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to(device)
        latents = 1 / vae.config.scaling_factor * latents
        image_tensor = vae.decode(latents).sample
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: vae = vae.to("cpu")

    image_tensor = (image_tensor.to("cpu") / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)

    final_svg_params = {
        'num_colors': None,
        'simplification_epsilon_factor': 0.002,
        'min_contour_area': 0.5,
        'max_features_to_render': 0
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
    guidance_scale=20,
    guidance_start_step=0,
    guidance_end_step=24,
    guidance_interval=1,
    guidance_resolution=256,
    device=device,
    seed=seed,
    lpips_mse_lambda=0.3,
    num_inference_steps=25,       # More steps can improve detail, 40-50 is a good balance.
    clip_guidance_scale=0.8,      # Strength of the semantic guidance. Adjust as needed.
    vector_guidance_scale=3.5,    # Strength of the SVG reconstruction guidance.
    # Use lower resolution memory-saving options if have a GPU with less VRAM
    use_half_precision=True,
    enable_sequential_cpu_offload=True,
    low_vram_shift_to_cpu=False
)
