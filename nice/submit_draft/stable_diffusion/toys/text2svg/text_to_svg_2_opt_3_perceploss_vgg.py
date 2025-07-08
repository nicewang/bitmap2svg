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

# 1. Import accelerate
import accelerate

# Import torchvision for VGG perceptual loss
import torchvision.models as models
import torchvision.transforms as transforms

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Differentiable SVG Renderer
import pydiffvg

import bitmap2svg

import logging

# stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v1-5/pytorch/default/2")
stable_diffusion_path = kagglehub.model_download("arishihu/stable-diffusion-v1-5/pytorch/default/2")

# --- Perceptual Loss Helper Class ---
class PerceptualLoss(torch.nn.Module):
    def __init__(self, device, feature_layers=[2, 7, 12, 21, 30]): # Corresponds to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 of VGG19
        super(PerceptualLoss, self).__init__()
        # Load pre-trained VGG19 features
        # We only use the 'features' part of VGG, not the classifier
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval().to("cpu") 
        self.feature_layers = feature_layers
        self.device = device
        
        # VGG expects images to be normalized with specific mean and std
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Move VGG to device only when needed (within forward pass)
        # It's crucial for memory management to keep it on CPU by default.
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img_gen, img_target, dtype):
        # Solution: Move VGG to the target device (GPU) AND cast its parameters to the specified dtype (e.g., torch.float16)
        # This ensures type consistency between input tensors and model parameters (like bias).
        self.vgg = self.vgg.to(self.device).to(dtype) # Explicitly cast VGG parameters to dtype

        # Clone and detach inputs to avoid modifying original tensors or backpropagating through them
        # Ensure inputs are in the correct range [0, 1] for VGG normalization
        img_gen_norm = self.normalize(img_gen.clamp(0, 1)).to(dtype)
        img_target_norm = self.normalize(img_target.clamp(0, 1)).to(dtype)
        
        loss = 0
        x_gen = img_gen_norm
        x_target = img_target_norm

        for i, layer in enumerate(self.vgg):
            x_gen = layer(x_gen)
            x_target = layer(x_target)
            if i in self.feature_layers:
                # Using L1 loss in feature space is often more robust than L2 for perceptual similarity
                loss += F.l1_loss(x_gen, x_target) # Using L1 loss on features
                # Alternative: loss += F.mse_loss(x_gen, x_target) # Using L2 loss on features

        # Move VGG back to CPU to free up GPU memory after computation
        self.vgg = self.vgg.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        
        return loss

# --- Original parse_svg_and_render function (no changes) ---
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
    perceptual_guidance_scale: float = 0.5, # NEW: Perceptual loss weight
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    guidance_resolution: int = 256,
    guidance_interval: int = 2,
    output_path: str = "output_guided_svg.svg",
    seed: int | None = None,
    enable_attention_slicing: bool = True,
    enable_cpu_offload: bool = True, # Note: this parameter is somewhat redundant with sequential_cpu_offload now
    use_half_precision: bool = True,
    batch_size: int = 1,  
    enable_sequential_cpu_offload: bool = True,
    low_vram_shift_to_cpu: bool = True # This is useful if accelerate's sequential offload isn't enough
):

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Restrict CUDA memory usage per process, adjust as needed
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Already set globally
    # torch.cuda.set_per_process_memory_fraction(0.90) # A slightly lower fraction for safety

    # --- Load Models with Optimized Memory Management ---
    dtype = torch.float16 if use_half_precision else torch.float32
    
    loading_kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,  
    }
    
    if use_half_precision:
        loading_kwargs["variant"] = "fp16"
    
    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **loading_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **loading_kwargs)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **loading_kwargs)
    except Exception as e:
        logging.warning(f"Failed to load with optimized settings, falling back to default: {e}")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Initialize Perceptual Loss outside the loop
    perceptual_criterion = PerceptualLoss(device=device) # Initialize on CPU
    
    if enable_attention_slicing:
        try:
            if hasattr(unet, 'enable_attention_slicing'):
                unet.enable_attention_slicing(1)  
            elif hasattr(unet, 'set_attention_slice'):
                unet.set_attention_slice(1)
        except Exception as e:
            logging.error(f"Could not enable attention slicing: {e}")
    
    if enable_sequential_cpu_offload:
        try:
            text_encoder = text_encoder.to("cpu")
            unet = unet.to("cpu") 
            vae = vae.to("cpu")
            
            unet = accelerate.cpu_offload(unet, execution_device=device)
            vae = accelerate.cpu_offload(vae, execution_device=device)
            text_encoder = accelerate.cpu_offload(text_encoder, execution_device=device)
            
        except Exception as e:
            logging.error(f"CPU offload failed, using manual device management: {e}")
            text_encoder = text_encoder.to("cpu")
            unet = unet.to("cpu")
            vae = vae.to("cpu")
            enable_sequential_cpu_offload = False
            low_vram_shift_to_cpu = True
    else:
        text_encoder = text_encoder.to("cpu")
        unet = unet.to("cpu")
        vae = vae.to("cpu")

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

    gc.collect()
    torch.cuda.empty_cache()

    height = 512
    width = 512
    
    with torch.no_grad():
        # Text encoder on device only during inference
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            text_encoder = text_encoder.to(device)
            
        text_input = tokenizer(
            [prompt], 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            text_encoder = text_encoder.to("cpu")
        
        uncond_input = tokenizer(
            [negative_prompt], 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        )
        
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            text_encoder = text_encoder.to(device)
            
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            text_encoder = text_encoder.to("cpu")
    
    gc.collect()
    torch.cuda.empty_cache()

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(device=device, dtype=dtype)
    
    del uncond_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    
    latent_height = int(height // 8)  # VAE scaling factor is 8
    latent_width = int(width // 8)
    latents = torch.randn(
        (batch_size, unet.config.in_channels, latent_height, latent_width),
        generator=generator, device=device, dtype=dtype
    )
    latents = latents * scheduler.init_noise_sigma
    
    scheduler.set_timesteps(num_inference_steps)
    
    svg_params_guidance = {
        'num_colors': 6,  
        'simplification_epsilon_factor': 0.02,  
        'min_contour_area': (guidance_resolution/512)**2 * 30.0,  
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
                pred_original_sample = scheduler.step(noise_pred_cfg, t, latents).pred_original_sample
                
                # VAE decoding for guidance
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
                    vae = vae.to(device)
                
                decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
                
                if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
                    vae = vae.to("cpu")
                
                # Resize for guidance resolution
                if guidance_resolution < height:
                    decoded_image_tensor_resized = F.interpolate(
                        decoded_image_tensor.float(),
                        size=(guidance_resolution, guidance_resolution),
                        mode='bilinear',  
                        align_corners=False
                    ).to(dtype)
                else:
                    decoded_image_tensor_resized = decoded_image_tensor.float()

                # Process for bitmap2svg
                img_to_vectorize_scaled = (decoded_image_tensor_resized / 2 + 0.5).clamp(0, 1)
                
                image_np = (img_to_vectorize_scaled.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                
                # Bitmap to SVG
                svg_string = bitmap2svg.bitmap_to_svg(pil_image, **svg_params_guidance)
                
                # Render SVG back to tensor
                rendered_svg_tensor = parse_svg_and_render(svg_string, pil_image.width, pil_image.height, device)
                rendered_svg_tensor_scaled = rendered_svg_tensor * 2.0 - 1.0 # Scale back to [-1, 1] range

                # --- Calculate Losses ---
                # Pixel-level MSE Loss
                pixel_loss = F.mse_loss(decoded_image_tensor_resized, rendered_svg_tensor_scaled)
                
                # Perceptual Loss (NEW)
                # decoded_image_tensor_resized is in [-1, 1], need to convert to [0, 1] for VGG
                perceptual_loss = perceptual_criterion(
                    (decoded_image_tensor_resized / 2 + 0.5), # Convert to [0, 1]
                    (rendered_svg_tensor_scaled / 2 + 0.5), # Convert to [0, 1]
                    dtype # Pass dtype to PerceptualLoss for type consistency
                )
                
                # Combine losses
                total_vector_loss = pixel_loss + perceptual_loss * perceptual_guidance_scale
                
                # Calculate gradient from combined loss
                # This part is crucial: we use the loss to compute a gradient that adjusts noise_pred_cfg
                # The gradient is derived from how much current image needs to change to reduce the loss
                grad = noise_pred_text - noise_pred_uncond # Original CFG gradient direction
                
                # Adjust noise_pred_cfg based on the new total_vector_loss
                # We're effectively pushing the noise prediction towards minimizing this loss
                # Note: loss.item() is used for scalar multiplication, as we don't backprop through the loss itself
                # We use the magnitude of the loss to scale the guidance vector
                noise_pred_cfg = noise_pred_cfg + (grad * total_vector_loss.item() * vector_guidance_scale)

                del pred_original_sample, decoded_image_tensor, rendered_svg_tensor, rendered_svg_tensor_scaled
                del img_to_vectorize_scaled, pil_image, svg_string
                
            gc.collect()
            torch.cuda.empty_cache()
        
        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Generating final image and SVG
    with torch.no_grad():
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            vae = vae.to(device)
        
        latents = 1 / vae.config.scaling_factor * latents
        image_tensor = vae.decode(latents).sample
        
        if low_vram_shift_to_cpu and not enable_sequential_cpu_offload:
            vae = vae.to("cpu")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    image_tensor = (image_tensor.to("cpu") / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    
    final_svg_params = {
        'num_colors': 24,  
        'simplification_epsilon_factor': 0.002, 
        'min_contour_area': 1.0,   
        'max_features_to_render': 0 
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image, **final_svg_params)
    
    return final_image, final_svg_string

if __name__ == '__main__':
    PROMPT = "vector illustration of a gray wool coat with a faux fur collar, flat design, solid colors, minimalist, clean lines"
    NEGATIVE_PROMPT = "photo, realistic, 3d, noisy, texture, blurry, shadow, gradient, complex details"

    NUM_STEPS = 27  
    VECTOR_GUIDANCE_SCALE = 2.0  
    PERCEPTUAL_GUIDANCE_SCALE = 0.5 # NEW: Adjust this weight
    GUIDANCE_START = int(NUM_STEPS * 0.1) 
    GUIDANCE_END = int(NUM_STEPS * 0.8)   
    
    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        perceptual_guidance_scale=PERCEPTUAL_GUIDANCE_SCALE, # Pass new parameter
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=256,  
        guidance_interval=3,      
        output_path="out_optimized.svg",
        seed=42,
        enable_attention_slicing=True,
        enable_cpu_offload=True, # This flag is somewhat redundant due to sequential_cpu_offload
        use_half_precision=True,
        enable_sequential_cpu_offload=True,
        low_vram_shift_to_cpu=True # Critical for low VRAM environments if sequential offload fails
    )

    print("Generation complete!")
