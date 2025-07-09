import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import re
import os
import random
import gc
import math

# Import necessary libraries
import accelerate
import lpips
import pydiffvg
import svgwrite 

# Set memory allocation strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import kagglehub

# Using SDv1.5
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

class ImprovedSvgGenerator(nn.Module):
    def __init__(self, num_paths=32, canvas_height=512, canvas_width=512, device="cpu"):
        super().__init__()
        self.device = device
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.num_paths = num_paths
        
        # Use more structured path initialization
        # Each path represents a simple shape (circle, ellipse, or polygon)
        self.centers = nn.Parameter(torch.rand(num_paths, 2) * 0.8 + 0.1)  # Center positions (0.1-0.9)
        self.radii = nn.Parameter(torch.rand(num_paths, 2) * 0.15 + 0.02)  # Radii for width/height
        self.rotations = nn.Parameter(torch.rand(num_paths) * 2 * math.pi)  # Rotation angles
        
        # Color parameters
        self.colors = nn.Parameter(torch.rand(num_paths, 4))  # RGBA
        
        # Path type (0=circle, 1=ellipse, 2=rectangle)
        self.path_types = nn.Parameter(torch.rand(num_paths))

    def _create_circle_path(self, center, radius, num_points=8):
        """Create a circle path using multiple points"""
        angles = torch.linspace(0, 2 * math.pi, num_points + 1, device=self.device)[:-1]
        points = torch.stack([
            center[0] + radius * torch.cos(angles),
            center[1] + radius * torch.sin(angles)
        ], dim=1)
        return points

    def _create_ellipse_path(self, center, radii, rotation, num_points=8):
        """Create an ellipse path with rotation"""
        angles = torch.linspace(0, 2 * math.pi, num_points + 1, device=self.device)[:-1]
        
        # Create ellipse points
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Apply radii
        x = radii[0] * cos_angles
        y = radii[1] * sin_angles
        
        # Apply rotation
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)
        
        rotated_x = x * cos_rot - y * sin_rot
        rotated_y = x * sin_rot + y * cos_rot
        
        points = torch.stack([
            center[0] + rotated_x,
            center[1] + rotated_y
        ], dim=1)
        return points

    def _create_rectangle_path(self, center, size, rotation):
        """Create a rectangle path with rotation"""
        half_w, half_h = size[0], size[1]
        
        # Rectangle corners
        corners = torch.tensor([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ], device=self.device)
        
        # Apply rotation
        cos_rot = torch.cos(rotation)
        sin_rot = torch.sin(rotation)
        
        rotated_corners = torch.stack([
            corners[:, 0] * cos_rot - corners[:, 1] * sin_rot,
            corners[:, 0] * sin_rot + corners[:, 1] * cos_rot
        ], dim=1)
        
        # Translate to center
        points = rotated_corners + center
        return points

    def forward(self):
        shapes, shape_groups = [], []
        
        for i in range(self.num_paths):
            # Convert normalized coordinates to canvas coordinates
            center = self.centers[i] * torch.tensor([self.canvas_width, self.canvas_height], device=self.device)
            radii = self.radii[i] * torch.tensor([self.canvas_width, self.canvas_height], device=self.device) * 0.5
            rotation = self.rotations[i]
            path_type = torch.sigmoid(self.path_types[i])
            
            # Determine path type and create appropriate points
            if path_type < 0.33:  # Circle
                radius = (radii[0] + radii[1]) / 2
                points = self._create_circle_path(center, radius)
            elif path_type < 0.66:  # Ellipse
                points = self._create_ellipse_path(center, radii, rotation)
            else:  # Rectangle
                points = self._create_rectangle_path(center, radii, rotation)
            
            # Create path with multiple segments
            num_points = points.shape[0]
            num_segments = num_points // 2
            
            # Create control points for smooth curves
            control_points = []
            for j in range(num_points):
                control_points.append(points[j])
                if j < num_points - 1:
                    # Add control point between current and next point
                    mid_point = (points[j] + points[(j + 1) % num_points]) / 2
                    control_points.append(mid_point)
            
            control_points = torch.stack(control_points)
            
            path = pydiffvg.Path(
                num_control_points=torch.tensor([2] * num_segments),
                points=control_points[:num_segments * 3],  # Limit points to avoid issues
                is_closed=True,
                stroke_width=torch.tensor(0.0)
            )
            
            shapes.append(path)
            
            # Apply color constraints
            color = torch.sigmoid(self.colors[i])  # Ensure colors are in [0,1]
            
            shape_groups.append(pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([i]),
                fill_color=color,
                stroke_color=None
            ))

        try:
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, self.canvas_height, shapes, shape_groups
            )
            img = pydiffvg.RenderFunction.apply(
                self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args
            )
            
            # Alpha compositing
            img = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * torch.ones(1, 1, 3, device=self.device)
            img = img.unsqueeze(0).permute(0, 3, 1, 2)
            
        except Exception as e:
            print(f"Rendering error: {e}")
            # Return white image if rendering fails
            img = torch.ones(1, 3, self.canvas_height, self.canvas_width, device=self.device)
            
        return img

def create_improved_svg_string(svg_generator, width, height):
    """Create SVG string with improved path generation"""
    dwg = svgwrite.Drawing(profile='tiny', size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
    
    with torch.no_grad():
        for i in range(svg_generator.num_paths):
            # Get parameters - ensure all tensors are on CPU
            center = (svg_generator.centers[i].cpu() * torch.tensor([width, height])).numpy()
            radii = (svg_generator.radii[i].cpu() * torch.tensor([width, height]) * 0.5).numpy()
            rotation = svg_generator.rotations[i].cpu().item()
            color = torch.sigmoid(svg_generator.colors[i]).cpu().numpy()
            path_type = torch.sigmoid(svg_generator.path_types[i]).cpu().item()
            
            # Create color string
            r, g, b, a = (np.clip(c, 0, 1) * 255 for c in color)
            color_hex = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            opacity = f"{a:.2f}"
            
            # Create path based on type with proper numeric formatting
            if path_type < 0.33:  # Circle
                radius = float((radii[0] + radii[1]) / 2)
                dwg.add(dwg.circle(
                    center=(float(center[0]), float(center[1])),
                    r=radius,
                    fill=color_hex,
                    fill_opacity=opacity,
                    stroke='none'
                ))
            elif path_type < 0.66:  # Ellipse
                dwg.add(dwg.ellipse(
                    center=(float(center[0]), float(center[1])),
                    r=(float(radii[0]), float(radii[1])),
                    fill=color_hex,
                    fill_opacity=opacity,
                    stroke='none',
                    transform=f'rotate({np.degrees(rotation):.1f} {center[0]:.1f} {center[1]:.1f})'
                ))
            else:  # Rectangle
                half_w, half_h = float(radii[0]), float(radii[1])
                dwg.add(dwg.rect(
                    insert=(float(center[0] - half_w), float(center[1] - half_h)),
                    size=(float(2 * half_w), float(2 * half_h)),
                    fill=color_hex,
                    fill_opacity=opacity,
                    stroke='none',
                    transform=f'rotate({np.degrees(rotation):.1f} {center[0]:.1f} {center[1]:.1f})'
                ))
    
    return dwg.tostring()

def generate_improved_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda:0",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    lpips_mse_lambda: float = 0.1,
    num_svg_paths: int = 32,  # Reduced for better optimization
    svg_optim_steps: int = 15,
    svg_lr: float = 0.02,
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    guidance_resolution: int = 512,  # Increased resolution
    guidance_interval: int = 1,
    output_path: str = "output_improved.svg",
    seed: int | None = None
):
    
    if seed is None: 
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Using seed: {seed}. Main device: {device}.")

    # Initialize components
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
    svg_generator = ImprovedSvgGenerator(num_svg_paths, guidance_resolution, guidance_resolution, device).to(device)
    
    # Use different learning rates for different parameters
    optimizer_svg = torch.optim.AdamW([
        {'params': [svg_generator.centers, svg_generator.radii], 'lr': svg_lr},
        {'params': [svg_generator.colors], 'lr': svg_lr * 0.5},
        {'params': [svg_generator.rotations, svg_generator.path_types], 'lr': svg_lr * 0.3}
    ], weight_decay=0.01)
    
    print("Loading models with sequential CPU offloading...")
    dtype = torch.float16
    loading_kwargs = {"torch_dtype": dtype, "use_safetensors": True, "low_cpu_mem_usage": True, "variant": "fp16"}
    
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **loading_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **loading_kwargs)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **loading_kwargs)
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("Enabling sequential CPU offload for all models...")
    unet = accelerate.cpu_offload(unet, execution_device=device)
    vae = accelerate.cpu_offload(vae, execution_device=device)
    text_encoder = accelerate.cpu_offload(text_encoder, execution_device=device)

    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xFormers enabled for UNet.")
    except (ImportError, AttributeError, Exception):
        print("xFormers not available. Using attention slicing.")
        if hasattr(unet, 'enable_attention_slicing'):
            unet.enable_attention_slicing()
            
    height, width = 512, 512
    
    # Prepare text embeddings
    with torch.no_grad():
        text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(device=device, dtype=dtype)
    del uncond_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    
    # Initialize latents
    latents = torch.randn((1, unet.config.in_channels, height//8, width//8), generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(num_inference_steps)
    
    print("Starting denoising and SVG optimization loop...")
    
    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # Diffusion step
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # SVG optimization step
        if guidance_start_step <= i < guidance_end_step and i % guidance_interval == 0:
            with torch.no_grad():
                # Get target image from current diffusion state
                pred_original_sample = scheduler.step(noise_pred_cfg, t, latents).pred_original_sample
                decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
                target_image = F.interpolate(decoded_image_tensor.float(), size=(guidance_resolution, guidance_resolution), mode='bilinear', align_corners=False)
                
            # Optimize SVG to match target
            for opt_step in range(svg_optim_steps):
                optimizer_svg.zero_grad()
                
                try:
                    rendered_svg = svg_generator()
                    rendered_svg_scaled = (rendered_svg.float() * 2.0 - 1.0)
                    
                    # Ensure tensors are on the same device
                    if rendered_svg_scaled.device != target_image.device:
                        rendered_svg_scaled = rendered_svg_scaled.to(target_image.device)
                    
                    # Multi-scale loss
                    loss_lpips_val = loss_fn_lpips(target_image, rendered_svg_scaled).mean()
                    loss_mse_val = F.mse_loss(target_image, rendered_svg_scaled)
                    
                    # Add regularization to prevent extreme parameter values
                    reg_loss = 0.001 * (torch.norm(svg_generator.centers) + torch.norm(svg_generator.radii))
                    
                    total_loss = loss_lpips_val + lpips_mse_lambda * loss_mse_val + reg_loss
                    
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(svg_generator.parameters(), max_norm=1.0)
                    
                    optimizer_svg.step()
                    
                    # Clamp parameters to valid ranges
                    with torch.no_grad():
                        svg_generator.centers.data.clamp_(0, 1)
                        svg_generator.radii.data.clamp_(0.01, 0.5)
                        
                except Exception as e:
                    print(f"SVG optimization error at step {opt_step}: {e}")
                    break
            
            del pred_original_sample, decoded_image_tensor, target_image
        
        # Update latents
        latents = scheduler.step(noise_pred_cfg, t, latents).prev_sample
        
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print("Generating final outputs...")
    
    # Generate final SVG
    svg_generator.to("cpu")
    svg_content = create_improved_svg_string(svg_generator, width, height)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Improved SVG saved to {output_path}")

    # Generate final raster image
    with torch.no_grad():
        latents = 1 / vae.config.scaling_factor * latents.to(device)
        image_tensor = vae.decode(latents).sample
        
    image_tensor = (image_tensor.to("cpu") / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    
    raster_output_path = os.path.splitext(output_path)[0] + ".png"
    final_image.save(raster_output_path)
    
    return final_image, svg_content

def gen_improved_bitmap(description, seed_val=42):
    """Generate improved bitmap with better SVG structure"""
    PROMPT = f"vector illustration of {description}, flat design, solid colors, minimalist, clean geometric shapes, simple forms"
    NEGATIVE_PROMPT = "photo, realistic, 3d, noisy, texture, blurry, shadow, gradient, complex details, photorealistic, ugly, deformed, chaotic"
    NUM_STEPS = 40
    GUIDANCE_START = int(NUM_STEPS * 0.1)  # Start guidance later
    GUIDANCE_END = int(NUM_STEPS * 0.8)    # End guidance earlier
    
    final_img, svg_content = generate_improved_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        model_id=stable_diffusion_path,
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        lpips_mse_lambda=0.05,  # Reduced MSE weight
        num_svg_paths=24,       # Reduced path count
        svg_optim_steps=12,     # Balanced optimization steps
        svg_lr=0.015,           # Adjusted learning rate
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=512,  # Higher resolution
        guidance_interval=1,
        seed=seed_val
    )
    
    return svg_content, final_img

# Example usage
if __name__ == "__main__":
    desc = "a purple silk scarf with tassel trim"
    svg, img = gen_improved_bitmap(desc, seed_val=42)