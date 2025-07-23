import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image, ImageOps
import numpy as np
import re
import os
import random
import gc

import kagglehub

# 1. Import necessary libraries
import accelerate
import lpips # Import the LPIPS library

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

# Differentiable SVG Renderer
import pydiffvg

import bitmap2svg

import logging

from torchvision import transforms

# print("Downloading Stable Diffusion model from Kaggle Hub...")
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
# print("Model download complete.")

def parse_svg_and_render(svg_string: str, width: int, height: int, device: str) -> torch.Tensor:
    polygons = re.findall(r'<polygon points="([^"]+)" fill="([^"]+)"/>', svg_string)
    shapes, shape_groups = [], []
    for points_str, fill_str in polygons:
        try:
            points_data = [float(p) for p in points_str.replace(',', ' ').split()]
            if not points_data or len(points_data) % 2 != 0: 
                continue

            points = torch.tensor(points_data, dtype=torch.float32, device=device).view(-1, 2)
            hex_color = fill_str.lstrip('#')
            if len(hex_color) == 3: 
                r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
            else: 
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

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
        if len(hex_color) == 3: 
            r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
        else: 
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_color_tensor = torch.tensor([r/255.0, g/255.0, b/255.0], device=device)

    if not shapes:
        return bg_color_tensor.view(1, 3, 1, 1).expand(1, 3, height, width)

    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(width, height, 2, 2, 0, None, *scene_args)
    img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * bg_color_tensor
    img = img.unsqueeze(0).permute(0, 3, 1, 2)

    return img

def create_latents_from_embedding(embedding: torch.Tensor, target_shape: tuple, generator: torch.Generator, vae: AutoencoderKL, device: str) -> torch.Tensor:
    """
    Creates initial latent tensor by first generating an image from text embedding, then encoding it through VAE.
    
    Args:
        embedding: Text embedding tensor (1, embedding_dim)
        target_shape: Target latent shape (batch_size, channels, height, width)
        generator: Random number generator for reproducibility
        vae: VAE model for encoding image to latent space
        device: Device to run computations on
    
    Returns:
        Latent tensor encoded through VAE with channel-wise normalization
    """
    # embedding shape is (1, embedding_dim), e.g., (1, 1024)
    # target_shape is (batch_size, channels, height, width), e.g., (1, 4, 64, 64)
    batch_size, channels, height, width = target_shape
    emb_dim = embedding.shape[1]
    
    # Calculate image dimensions from latent dimensions
    image_height = height * 8  # VAE downsamples by factor of 8
    image_width = width * 8
    
    # Create structured image from embedding
    # Use embedding to create spatial patterns
    chunk_size = emb_dim // 3  # Split into RGB channels
    
    if chunk_size == 0:
        # Fallback: create random image and encode it
        random_image = torch.randn(batch_size, 3, image_height, image_width, generator=generator, device=device)
        random_image = torch.tanh(random_image)  # Normalize to [-1, 1]
        
        with torch.no_grad():
            latent = vae.encode(random_image).latent_dist.sample(generator=generator)
            latent = latent * vae.config.scaling_factor
        
        return latent
    
    # Calculate spatial dimensions for embedding reshaping
    spatial_size = int(np.sqrt(chunk_size))
    if spatial_size == 0:
        spatial_size = 1
    
    # Create RGB channels from embedding
    rgb_channels = []
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, emb_dim)
        channel_data = embedding[0, start_idx:end_idx]
        
        # Pad if necessary
        needed_size = spatial_size * spatial_size
        if len(channel_data) < needed_size:
            padding = torch.randn(needed_size - len(channel_data), device=device, generator=generator)
            channel_data = torch.cat([channel_data, padding])
        else:
            channel_data = channel_data[:needed_size]
        
        # Reshape to spatial dimensions
        channel_image = channel_data.view(1, 1, spatial_size, spatial_size)
        
        # Interpolate to target image size
        channel_image = F.interpolate(channel_image, size=(image_height, image_width), mode='bilinear', align_corners=False)
        rgb_channels.append(channel_image)
    
    # Combine RGB channels
    structured_image = torch.cat(rgb_channels, dim=1)  # Shape: (1, 3, image_height, image_width)
    
    # Normalize to [-1, 1] range (expected by VAE)
    if structured_image.std() > 0:
        structured_image = (structured_image - structured_image.mean()) / structured_image.std()
    structured_image = torch.tanh(structured_image)  # Ensure [-1, 1] range
    
    # Encode through VAE to get latent representation
    with torch.no_grad():
        latent = vae.encode(structured_image).latent_dist.sample(generator=generator)
        latent = latent * vae.config.scaling_factor
    
    # Apply channel-wise normalization to match training distribution
    for c in range(latent.shape[1]):  # Iterate over channels
        channel = latent[:, c:c+1, :, :]
        # Normalize each channel to N(0,1) distribution
        channel_mean = channel.mean()
        channel_std = channel.std()
        if channel_std > 1e-8:  # Avoid division by zero
            latent[:, c:c+1, :, :] = (channel - channel_mean) / channel_std
    
    noise_ratio = 0.95
    noise = torch.randn_like(latent)
    # Mix the original structured latent with random noise
    latent = latent * (1 - noise_ratio) + noise * noise_ratio

    return latent


# def get_progressive_guidance_config(current_step: int, total_steps: int) -> dict:
#     """
#     Get progressive guidance configuration based on current step.
    
#     Args:
#         current_step: Current denoising step
#         total_steps: Total number of inference steps
    
#     Returns:
#         Dictionary containing guidance weights for current stage
#     """
#     # progress = current_step / total_steps
    
#     # # Early stage (0-30%): Focus on semantic structure
#     # if progress < 0.3:
#     #     return {
#     #         'clip_weight': 1.0,      # Semantic matching is most important
#     #         'lpips_weight': 0.3,     # Lower perceptual loss
#     #         'mse_weight': 0.1,       # Lowest MSE
#     #         'guidance_strength': 0.8  # Strong guidance
#     #     }
#     # # Middle stage (30-70%): Balance perceptual quality
#     # elif progress < 0.7:
#     #     return {
#     #         'clip_weight': 0.7,      # Maintain semantic
#     #         'lpips_weight': 0.8,     # Improve perceptual quality
#     #         'mse_weight': 0.3,       # Moderate MSE
#     #         'guidance_strength': 0.6  # Medium guidance
#     #     }
#     # # Late stage (70-100%): Detail refinement
#     # else:
#     #     return {
#     #         'clip_weight': 0.5,      # Lower semantic weight
#     #         'lpips_weight': 1.0,     # Highest perceptual loss
#     #         'mse_weight': 0.5,       # Detail optimization
#     #         'guidance_strength': 0.4  # Light guidance
#     #     }
    
#     progress = current_step / total_steps
    
#     if progress < 0.2:
#         current_clip_scale = 0.05 
#     elif progress < 0.8:
#         current_clip_scale = np.sin((progress - 0.2) / 0.6 * np.pi) 
#     else:
#         current_clip_scale = 0.2
        
#     return {'clip_weight': current_clip_scale}

def get_progressive_guidance_config(current_step: int, total_steps: int) -> dict:
    
    progress = current_step / total_steps
    
    if progress < 0.15:
        current_clip_scale = 0.0
        
    elif progress < 0.50:
        current_clip_scale = (progress - 0.15) / (0.50 - 0.15)
        
    elif progress < 0.90:
        progress_in_decay = (progress - 0.50) / (0.90 - 0.50)
        current_clip_scale = 0.5 * (1 + np.cos(np.pi * progress_in_decay))
        
    else:
        current_clip_scale = 0.0
    
    current_mse_weight = get_reconstruction_loss_lambda(progress)
    current_reconstruction_weight = get_reconstruction_guidance_scale(progress)

    return {
        'clip_weight': current_clip_scale,
        'mse_weight': current_mse_weight,
        'reconstruct_weight': current_reconstruction_weight
    }

def get_reconstruction_guidance_scale(progress: float, start_scale: float = 0.2, end_scale: float = 1.0) -> float:

    if progress < 0.0:
        progress = 0.0
    if progress > 1.0: 
        progress = 1.0

    current_scale = start_scale + (end_scale - start_scale) * progress
    return current_scale

def get_reconstruction_guidance_scale_v2(progress: float, start_scale: float = 0.1, end_scale: float = 1.2, power: float = 2.0) -> float:

    if progress < 0.0: 
        progress = 0.0
    if progress > 1.0: 
        progress = 1.0
        
    current_scale = start_scale + (end_scale - start_scale) * (progress ** power)
    return current_scale

def get_reconstruction_loss_lambda(progress: float, max_lambda: float = 0.3) -> float:
    
    if progress < 0.5:
        return 0.0
    else:
        return max_lambda * ((progress - 0.5) / 0.5)

import numpy as np

def get_universal_guidance_configs(t: int, clip_max_scale: float = 1.1) -> dict:

    configs = {
        'clip_scale': 0.0,
        'recon_scale': 0.0,
        'recon_lambda': 0.0,
    }

    if t > 700:
        configs['clip_scale'] = 0.9
        configs['recon_scale'] = 0.2
        configs['recon_lambda'] = 0.0
        
    elif t > 200:
        progress_in_phase = (700 - t) / 500
        configs['clip_scale'] = 1.0 - 0.5 * progress_in_phase
        configs['recon_scale'] = 0.2 + 0.8 * progress_in_phase
        configs['recon_lambda'] = 0.4 * progress_in_phase

    else:
        progress_in_phase = (200 - t) / 200
        configs['clip_scale'] = 0.5 * (1 - progress_in_phase)
        configs['recon_scale'] = 1.0
        configs['recon_lambda'] = 0.4

    configs['clip_scale'] *= clip_max_scale
    
    return configs

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

    # model_id = stable_diffusion_path
    
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
    
    # print(f"Initializing LPIPS model on {guidance_device}...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(guidance_device).eval()

    # print(f"Initializing CLIP model and processor on {guidance_device}...")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(guidance_device).eval()
    
    dtype = torch.float16 if use_half_precision else torch.float32
    loading_kwargs = {"torch_dtype": dtype, "use_safetensors": True, "low_cpu_mem_usage": True}
    if use_half_precision: 
        loading_kwargs["variant"] = "fp16"

    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **loading_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **loading_kwargs)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **loading_kwargs)
    except Exception as e:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

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

    gc.collect()
    torch.cuda.empty_cache()

    # --- Input Preparation ---
    height = 512
    width = 512

    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
            text_encoder = text_encoder.to(device)
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        prompt_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        if description != "":
            des_input = tokenizer([description], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            input_embeddings = text_encoder(des_input.input_ids.to(device))[0]
        else:
            input_embeddings = prompt_embeddings
        
        uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
            text_encoder = text_encoder.to("cpu")
        
        text_embeddings = torch.cat([uncond_embeddings, prompt_embeddings]).to(device=device, dtype=dtype)
        del uncond_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        if description != "":
            clip_text_input = clip_processor(text=[description], return_tensors="pt", padding=True).to(guidance_device)
        else:
            clip_text_input = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(guidance_device)
        text_features = clip_model.get_text_features(**clip_text_input)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    latent_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)
    
    # Move VAE to device temporarily for encoding
    if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
        vae = vae.to(device)
    
    structured_latents = create_latents_from_embedding(input_embeddings.mean(dim=1), latent_shape, generator, vae, device)
    
    # Move VAE back to CPU if needed
    if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
        vae = vae.to("cpu")
    
    # Create pure random noise
    random_latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    
    # Blend the structured and random latents based on the strength parameter
    latents = (1.0 - strength) * structured_latents + strength * random_latents
    

    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    svg_params_guidance = {
        'num_colors': None,
        'simplification_epsilon_factor': 0.01,
        'min_contour_area': (guidance_resolution/512)**2 * 10.0,
        'max_features_to_render': 64
    }

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        if i % 5 == 0: 
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
                unet = unet.to(device)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
                unet = unet.to("cpu")

        del latent_model_input
        gc.collect()
        torch.cuda.empty_cache()

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if guidance_start_step <= i < guidance_end_step and i % guidance_interval == 0:
            with torch.no_grad():

                # # ========== DDIMScheduler ==========
                # pred_original_sample = scheduler.step(noise_pred_cfg, t, latents).pred_original_sample
                # # ========== DDIMScheduler End ==========

                # ========== DPMSolverMultistepScheduler ==========
                # Create temporary scheduler to avoid state corruption
                temp_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
                temp_scheduler.set_timesteps(num_inference_steps)
                
                # For DPMSolverMultistepScheduler, we need to handle the step differently
                temp_step_result = temp_scheduler.step(noise_pred_cfg, t, latents)
                if hasattr(temp_step_result, 'pred_original_sample'):
                    pred_original_sample = temp_step_result.pred_original_sample
                else:
                    # Fallback: use current latents as approximation
                    pred_original_sample = latents
                # ========== DPMSolverMultistepScheduler End ==========
                
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
                    vae = vae.to(device)
                decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
                    vae = vae.to("cpu")

                target_image_for_loss = decoded_image_tensor
                if guidance_resolution < height:
                    target_image_for_loss = F.interpolate(decoded_image_tensor.float(), size=(guidance_resolution, guidance_resolution), mode='bilinear', align_corners=False).to(dtype)

                img_to_vectorize_scaled = (target_image_for_loss / 2 + 0.5).clamp(0, 1)
                image_np = (img_to_vectorize_scaled.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image_posterized = ImageOps.posterize(pil_image, 4)
                svg_string = bitmap2svg.bitmap_to_svg(pil_image_posterized, **svg_params_guidance)
                rendered_svg_tensor = parse_svg_and_render(svg_string, pil_image.width, pil_image.height, device)
                rendered_svg_tensor_scaled = rendered_svg_tensor * 2.0 - 1.0

                # Switch to guidance device for loss calculation
                target_image_gpu = target_image_for_loss.to(guidance_device)
                rendered_svg_gpu = rendered_svg_tensor_scaled.to(guidance_device)

                loss_lpips_val = loss_fn_lpips(target_image_gpu.float(), rendered_svg_gpu.float()).mean()
                loss_mse_val = F.mse_loss(target_image_gpu, rendered_svg_gpu)

                soften_augment_transforms = transforms.Compose([
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    # transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
                ])

                softened_svg_for_clip = soften_augment_transforms(rendered_svg_tensor)

                noise_level = 0.03 
                # noise_level = 0.02
                noise = torch.randn_like(softened_svg_for_clip) * noise_level
                softened_svg_for_clip = (softened_svg_for_clip + noise).clamp(0, 1)

                clip_image_input = clip_processor(images=softened_svg_for_clip.to(guidance_device), return_tensors="pt").to(guidance_device)
                # clip_image_input = clip_processor(images=rendered_svg_tensor, return_tensors="pt").to(guidance_device)
                
                image_features = clip_model.get_image_features(**clip_image_input)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                clip_loss = 1 - (text_features @ image_features.T).squeeze()
                
                # # # Get progressive guidance configuration
                # guidance_config = get_progressive_guidance_config(i, num_inference_steps)
                
                # # # Apply progressive weights to losses
                # # weighted_lpips_loss = guidance_config['lpips_weight'] * loss_lpips_val
                # weighted_mse_loss = guidance_config['mse_weight'] * loss_mse_val
                # weighted_clip_loss = guidance_config['clip_weight'] * clip_loss
                
                # # Calculate final losses
                # # reconstruction_loss = weighted_lpips_loss + weighted_mse_loss
                # # reconstruction_loss = loss_lpips_val + lpips_mse_lambda * loss_mse_val
                # reconstruction_loss = guidance_config['reconstruct_weight'] * (loss_lpips_val + weighted_mse_loss)
                # total_loss = reconstruction_loss + clip_guidance_scale * weighted_clip_loss
                # # total_loss = reconstruction_loss + clip_guidance_scale * clip_loss
                
                # # # Apply progressive guidance strength
                # # progressive_vector_guidance_scale = vector_guidance_scale * guidance_config['guidance_strength']

                guidance_configs = get_universal_guidance_configs(t, clip_max_scale=1.1)

                reconstruction_loss = loss_lpips_val + guidance_configs['recon_lambda'] * loss_mse_val
                scaled_reconstruction_loss = reconstruction_loss * guidance_configs['recon_scale']
                
                clip_loss_term = guidance_configs['clip_scale'] * clip_loss

                total_loss = scaled_reconstruction_loss + clip_loss_term

                grad = noise_pred_text - noise_pred_uncond
                # noise_pred_cfg = noise_pred_cfg + (grad * total_loss.item() * progressive_vector_guidance_scale)
                noise_pred_cfg = noise_pred_cfg + (grad * total_loss.item() * vector_guidance_scale)


            gc.collect(); 
            torch.cuda.empty_cache()

        # Use scheduler step properly for DPMSolverMultistepScheduler
        step_result = scheduler.step(noise_pred_cfg, t, latents)
        if hasattr(step_result, 'prev_sample'):
            latents = step_result.prev_sample
        else:
            latents = step_result

    # --- Final Generation (Preserved) ---
    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
            vae = vae.to(device)
        latents = 1 / vae.config.scaling_factor * latents
        image_tensor = vae.decode(latents).sample
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload: 
            vae = vae.to("cpu")

    image_tensor = (image_tensor.cpu() / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    final_image_posterized = ImageOps.posterize(final_image, 4)

    final_svg_params = {
        'num_colors': None, 
        'simplification_epsilon_factor': 0.002,
        'min_contour_area': 0.1, 
        'max_features_to_render': 0
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image_posterized, **final_svg_params)
    
    del vae, text_encoder, unet, tokenizer, scheduler, loss_fn_lpips, clip_model, clip_processor
    gc.collect()
    torch.cuda.empty_cache()

    return final_image, final_svg_string


# prompt = "A cute cat wearing a wizard hat, minimalist logo, vector art"
# negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render"

# # prompt = "gray wool coat with a faux fur collar, vector art"
# prompt = "simple vector illustration of gray wool coat with a faux fur collar"
# negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render"

description = "orange corduroy overalls"
# prompt = (
#     f"Simple vector illustration of {description} "
#     "with flat color blocks, beautiful, minimal details, solid colors only"
# )
# negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"

# prompt = f"simple vector illustration of {description}"
# negative_prompt = "blurry, pixelated, jpeg artifacts, low quality, photorealistic, complex, 3d render, lines, framing, hatching, background, textures, patterns, details, outlines"

prompt = (
    f"clean classic vector illustration of {description}, on a light white background, "
    "edge-to-edge scene, no borders, flat design, solid colors only, minimalist, simple shapes, "
    "geometric style, flat color blocks, minimal details, no complex details"
)

# "lines, framing, hatching, background, patterns, outlines, "
negative_prompt = (
    "photo, realistic, 3d, noisy, textures, blurry, shadow, "
    "framing, border, circle, vignette, cropped, picture-in-picture, "
    "lines, hatching, background, outlines, "
    "gradient, complex details, patterns, stripes, dots, "
    "repetitive elements, small details, intricate designs, "
    "photo frame, emblem, logo, contained within a shape, "
    "busy composition, cluttered"
)

seed = 42
device = "cuda:0"

img, svg = generate_svg_with_guidance(
    prompt=prompt,
    negative_prompt=negative_prompt,
    description=description,
    device=device,
    # --- Strength parameter for blending structured and random noise ---
    strength=1.0, # 1.0 is pure noise, 0.0 is pure structure
    num_inference_steps=15,
    guidance_scale=20,
    vector_guidance_scale=4.5,
    # ToDo: parameter adjustment
    lpips_mse_lambda=0.1, 
    clip_guidance_scale=0.0, 
    # ToDo-End: parameter adjustment
    guidance_start_step=0,
    guidance_end_step=15,
    guidance_resolution=1024,
    guidance_interval=1,
    seed=42,
    use_half_precision=True,
    enable_sequential_cpu_offload=True,
    low_vram_shift_to_cpu=False
)
