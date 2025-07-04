import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import pynvml
import random
import re
import os

# Diffusers and Transformers
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Differentiable SVG Renderer
import diffvg

import bitmap2svg

# --- Step 1: Helper function for Differentiable Rendering ---
def parse_svg_and_render(svg_string: str, width: int, height: int) -> torch.Tensor:
    """
    Parses the SVG string to extract polygon data and renders it using diffvg.
    NOTE: This is a simplified parser for <polygon> tags.
    output structure might require a more robust parser (e.g., using xml.etree).
    """
    # Use regex to find all polygon tags and their data
    polygons = re.findall(r'<polygon points="([^"]+)" fill="([^"]+)"/>', svg_string)
    
    shapes = []
    shape_groups = []

    for points_str, fill_str in polygons:
        # Parse points
        points = [float(p) for p in points_str.replace(',', ' ').split()]
        points = torch.tensor(points, dtype=torch.float32).view(-1, 2)
        
        # Parse color
        # diffvg uses [r, g, b, a] in range [0, 1]
        hex_color = fill_str.lstrip('#')
        if len(hex_color) == 3: # Handle compressed hex #RGB
            r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
        else: # Handle #RRGGBB
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        color = torch.tensor([r / 255.0, g / 255.0, b / 255.0, 1.0])
        
        path = diffvg.Polygon(points=points, is_closed=True)
        shapes.append(path)
        # diffvg requires a shape group for each path with its own fill color
        shape_groups.append(diffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=color))

    # Find the background color from the <rect> tag
    bg_match = re.search(r'<rect .* fill="([^"]+)"/>', svg_string)
    bg_color = None
    if bg_match:
        hex_color = bg_match.group(1).lstrip('#')
        if len(hex_color) == 3:
            r, g, b = tuple(int(hex_color[i]*2, 16) for i in range(3))
        else:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_color = torch.tensor([r / 255.0, g / 255.0, b / 255.0, 1.0])


    # Set up the scene and render
    scene_args = diffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    render = diffvg.RenderFunction.apply
    img = render(width, height, 2, 2, 0, None, *scene_args) # 2x2 MSAA
    
    # Premultiplied alpha -> standard alpha
    img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * (bg_color if bg_color is not None else 0.0)
    
    # Reshape to PyTorch format (C, H, W) and add batch dimension
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    return img

def sds_loss(latents, vae, scheduler, unet, text_embeddings, t, guidance_scale, vector_guidance_scale, svg_params):
    """
    Calculates the Score Distillation Sampling loss for vector guidance.
    """
    # 1. Enable gradients on latents
    latents = latents.detach().requires_grad_(True)
    
    # 2. Predict noise (same as in the main loop)
    latent_model_input = torch.cat([latents] * 2) # For CFG
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # 3. Predict the "clean" image (x0_pred) from the noisy latents
    # We use the scheduler to "undo" one step of diffusion
    pred_original_sample = scheduler.step(noise_pred, t, latents).pred_original_sample

    # 4. Decode the predicted clean latents into an image
    # The VAE output is in the range [-1, 1]
    decoded_image_tensor = vae.decode(1 / vae.config.scaling_factor * pred_original_sample).sample
    
    # 5. Convert tensor to PIL Image for the C++ library
    # Clamp and scale from [-1, 1] to [0, 255]
    decoded_image_tensor_scaled = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)
    # (B, C, H, W) -> (B, H, W, C)
    image_np = (decoded_image_tensor_scaled.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    
    # 6. Call bitmap2svg to get the SVG
    svg_string = bitmap2svg.bitmap_to_svg(pil_image, **svg_params)
    
    # 7. Differentiably render the SVG back to a tensor image
    # This tensor will be in the range [0, 1]
    rendered_svg_tensor = parse_svg_and_render(svg_string, pil_image.width, pil_image.height)
    # Match the device and range of the VAE output
    rendered_svg_tensor = rendered_svg_tensor.to(decoded_image_tensor.device)
    rendered_svg_tensor_scaled = rendered_svg_tensor * 2 - 1 # from [0, 1] to [-1, 1]

    # 8. Calculate the guidance loss
    # We compare the image SD *wants* to make with the image our vectorizer *can* make
    loss = F.mse_loss(decoded_image_tensor, rendered_svg_tensor_scaled)
    
    # 9. Backpropagate the loss to get gradients on the latents
    grad = torch.autograd.grad(loss, latents, grad_outputs=torch.ones_like(loss))[0]
    
    # The gradient is scaled to prevent it from being too powerful
    # This is a common practice in guidance techniques
    grad = grad * vector_guidance_scale
    
    # Can experiment with different ways to scale the gradient
    # target = latents - grad
    # loss = F.mse_loss(latents, target) * vector_guidance_scale
    # loss.backward()
    
    # Return the computed gradient
    return grad

# --- Step 2: Main Generation Function with the Optimization Loop ---
def generate_svg_with_guidance(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    vector_guidance_scale: float = 5.0,
    guidance_start_step: int = 10, # Start guidance after initial structure forms
    guidance_end_step: int = 40,   # Stop guidance before the final steps
    output_path: str = "output_guided_svg.svg",
    seed: int | None = None
):
    """
    Main function to generate an SVG from a text prompt using iterative guidance.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.manual_seed(seed)
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
    height = unet.config.sample_size * vae.config.scale_factor
    width = unet.config.sample_size * vae.config.scale_factor
    
    # Text embeddings
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    uncond_input = tokenizer([negative_prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # For Classifier-Free Guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initial random latents
    latents = torch.randn(
        (1, unet.config.in_channels, height // vae.config.scale_factor, width // vae.config.scale_factor),
        generator=generator,
        device=device
    )
    latents = latents * scheduler.init_noise_sigma
    
    # --- C. Denoising and Guidance Loop ---
    print("Starting denoising and guidance loop...")
    scheduler.set_timesteps(num_inference_steps)
    
    # Parameters for C++ vectorizer, passed to the guidance loss function
    svg_params = {
        'num_colors': 12,
        'simplification_epsilon_factor': 0.01,
        'min_contour_area': 15.0,
        'max_features_to_render': 256 # Limit features during guidance for speed
    }

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        # --- Standard Denoising Step ---
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # --- Vector Guidance Step ---
        if guidance_start_step <= i < guidance_end_step:
            # Calculate the guidance gradient
            grad = sds_loss(latents, vae, scheduler, unet, text_embeddings, t, guidance_scale, vector_guidance_scale, svg_params)
            
            # Apply the gradient to the latents
            # The scaling factor here (alpha) is important for stability
            alpha_prod_t = scheduler.alphas_cumprod[t]
            grad_scale = (1 - alpha_prod_t)**0.5 # Scale similarly to original noise
            latents = latents - grad * grad_scale
        
        # --- Update Latents for Next Step ---
        # Compute the previous noisy sample x_t -> x_{t-1}
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
    # --- D. Final Image and SVG Generation ---
    print("Generating final image and SVG...")
    # Scale and decode the final latents
    latents = 1 / vae.config.scaling_factor * latents
    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    # Convert to PIL Image
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    final_image = Image.fromarray(image_np)
    
    # Save the final raster image
    raster_output_path = os.path.splitext(output_path)[0] + ".png"
    final_image.save(raster_output_path)
    print(f"Saved final raster image to: {raster_output_path}")

    # Perform one final, high-quality vectorization on the result
    final_svg_params = {
        'num_colors': 16,
        'simplification_epsilon_factor': 0.005,
        'min_contour_area': 5.0,
        'max_features_to_render': 0 # Unlimited
    }
    final_svg_string = bitmap2svg.bitmap_to_svg(final_image, **final_svg_params)
    
    with open(output_path, "w") as f:
        f.write(final_svg_string)
    print(f"Saved final SVG to: {output_path}")

    return final_image, final_svg_string

# --- Example Usage ---
if __name__ == '__main__':    
    # Check for GPU and memory
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
    PROMPT = "A beautiful stained glass window of a roaring lion's head, minimalist, vector logo"
    NEGATIVE_PROMPT = "photo, realistic, 3d, blurry, noisy, text, watermark, signature, jpeg artifacts"
    
    # Lower steps for faster testing, increase to 50 for quality
    NUM_STEPS = 50
    
    # Strength of the vector guidance. Higher values force a more "vector-like" style.
    # Start with a low value and increase.
    VECTOR_GUIDANCE_SCALE = 8.0 
    
    # When to apply the guidance.
    # Don't start too early (let the main shape form) or end too late (let it refine details).
    GUIDANCE_START = int(NUM_STEPS * 0.1)
    GUIDANCE_END = int(NUM_STEPS * 0.8)

    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        device=DEVICE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        output_path="lion_logo.svg",
        seed=42 # Use a fixed seed for reproducibility
    )
