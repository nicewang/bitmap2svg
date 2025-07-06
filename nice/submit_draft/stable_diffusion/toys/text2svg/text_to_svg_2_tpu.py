import jax
import jax.numpy as jnp

# ==============================================================================
# EXPLICIT JAX DISTRIBUTED INITIALIZATION
# This is done at the very beginning to ensure JAX correctly configures
# itself for the managed TPU environment (like Kaggle), preventing low-level
# networking and initialization errors.
# ==============================================================================
try:
    print("Initializing JAX distributed environment...")
    jax.distributed.initialize()
    print(f"JAX initialized. Found {jax.device_count()} devices.")
except Exception as e:
    print(f"Warning: Could not initialize JAX distributed environment: {e}")
    print(f"Proceeding with {jax.device_count()} local devices.")


import numpy as np
import cv2  # OpenCV for vectorization logic
import cairosvg # For rendering SVG on CPU
from PIL import Image
from tqdm.auto import tqdm
import os
import random
import gc

from diffusers import FlaxStableDiffusionPipeline, FlaxDDIMScheduler
from transformers import CLIPTokenizer
from flax.jax_utils import replicate
from flax.training.common_utils import shard

# -----------------------------------------------------------------------------------
# Part 1: Host-side (CPU) Vectorization and Rendering Logic
# This part remains unchanged. It runs on the host CPU.
# -----------------------------------------------------------------------------------

def host_vectorize_and_render(
    image_np: np.ndarray,
    num_colors: int,
    simplification_epsilon_factor: float,
    min_contour_area: float,
    max_features: int
) -> tuple[np.ndarray, str]:
    """
    Performs vectorization on the host CPU. This function takes a numpy image,
    converts it to SVG, and renders it back to a numpy image for loss calculation.
    Args:
        image_np: Input image as a NumPy array [0, 1] range, shape (H, W, 3).
    Returns:
        A tuple containing:
        - rendered_image_np: The SVG rendered back to a NumPy array [0, 1].
        - svg_string: The generated SVG string.
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Input image must be in HWC format.")

    height, width, _ = image_np.shape
    if height == 0 or width == 0:
        print("Warning: host_vectorize_and_render received an empty image.")
        return np.zeros_like(image_np), "<svg></svg>"
    
    # 1. Convert to uint8 for OpenCV processing
    img_uint8 = (image_np * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

    # 2. Color Quantization using k-means
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    palette = [tuple(map(int, color)) for color in centers] # BGR palette

    # 3. Find contours for each color
    all_polygons = []
    quantized_img_bgr = centers[labels.flatten()].reshape(img_rgb.shape)
    for color_bgr in palette:
        mask_color = cv2.inRange(quantized_img_bgr, np.array(color_bgr), np.array(color_bgr))
        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hex_color = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            epsilon = simplification_epsilon_factor * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_poly) >= 3:
                all_polygons.append({
                    "points": approx_poly.reshape(-1, 2),
                    "color": hex_color,
                    "area": cv2.contourArea(contour)
                })

    all_polygons.sort(key=lambda p: p["area"], reverse=True)
    if max_features > 0:
        all_polygons = all_polygons[:max_features]

    # 5. Build SVG string
    avg_color_bgr = np.mean(pixels, axis=0).astype(int)
    bg_hex_color = f"#{avg_color_bgr[2]:02x}{avg_color_bgr[1]:02x}{avg_color_bgr[0]:02x}"

    svg_elements = [f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>']
    for poly in all_polygons:
        points_str = " ".join([f"{p[0]},{p[1]}" for p in poly["points"]])
        svg_elements.append(f'<polygon points="{points_str}" fill="{poly["color"]}"/>')
    
    svg_string = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(svg_elements)}</svg>'

    # 6. Render SVG back to a bitmap using CairoSVG
    try:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        rendered_img_np = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)
        rendered_img_np = cv2.cvtColor(rendered_img_np, cv2.COLOR_BGR2RGB) # back to RGB
        rendered_img_np = rendered_img_np.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Warning: SVG rendering failed: {e}. Returning a black image.")
        rendered_img_np = np.zeros_like(image_np, dtype=np.float32)

    return rendered_img_np, svg_string

# -----------------------------------------------------------------------------------
# Part 2: JAX/TPU Accelerated Generation Function (Optimized Version)
# -----------------------------------------------------------------------------------

def generate_svg_with_guidance_tpu(
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    dtype: jnp.dtype = jnp.bfloat16, # bfloat16 is optimal for TPUs
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    vector_guidance_scale: float = 1.5,
    guidance_start_step: int = 5,
    guidance_end_step: int = 40,
    guidance_resolution: int = 256,
    guidance_interval: int = 2,
    output_path: str = "output_guided_svg_tpu.svg",
    seed: int | None = None,
):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    num_devices = jax.device_count()
    print(f"Using prompt: '{prompt}'")
    print(f"Using seed: {seed}. Running on {num_devices} TPU devices.")

    # 1. Load models and necessary components
    print("Loading tokenizer and scheduler...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = FlaxDDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("Loading Stable Diffusion pipeline to extract parameters...")
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        model_id, revision="flax", dtype=dtype
    )
    
    # ==============================================================================
    # PROACTIVE HOST MEMORY MANAGEMENT
    # After extracting the parameters, the large pipeline object is no longer
    # needed in host RAM. We delete it to prevent Out-of-Memory crashes.
    # ==============================================================================
    image_height, image_width = pipeline.height, pipeline.width
    
    print("Replicating parameters to all TPU devices...")
    params = replicate(params)

    print("Deleting large pipeline object to free host RAM...")
    del pipeline
    gc.collect()

    # 2. Prepare inputs
    prompts = [prompt] * num_devices
    neg_prompts = [negative_prompt] * num_devices
    
    prompt_ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np").input_ids
    uncond_prompt_ids = tokenizer(neg_prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np").input_ids
    
    # Shard inputs for pmap
    prompt_ids = shard(prompt_ids)
    uncond_prompt_ids = shard(uncond_prompt_ids)
    
    # 3. Define pmap-ed functions for TPU execution
    # These functions are now self-contained and accept model parameters explicitly.
    
    @jax.pmap
    def pmap_prepare_text_embeddings(prompt_ids, uncond_prompt_ids, text_encoder_params):
        text_encoder = FlaxCLIPTextModel(config=CLIPTextConfig.from_pretrained(f"{model_id}/text_encoder"))
        
        uncond_embeddings = text_encoder(input_ids=uncond_prompt_ids, params=text_encoder_params)[0]
        text_embeddings = text_encoder(input_ids=prompt_ids, params=text_encoder_params)[0]
        return jnp.concatenate([uncond_embeddings, text_embeddings])

    @jax.pmap
    def pmap_predict_noise(latents, text_embeddings, t, unet_params):
        latent_model_input = jnp.concatenate([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = UNet2DConditionModel.from_config(f"{model_id}/unet").apply(
            {'params': unet_params}, latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample
        return noise_pred

    @jax.pmap
    def pmap_get_pred_original_sample(noise_pred, t, latents):
        return scheduler.step(noise_pred, t, latents).pred_original_sample

    @jax.pmap
    def pmap_scheduler_step(noise_pred, t, latents):
        return scheduler.step(noise_pred, t, latents).prev_sample

    @jax.pmap
    def pmap_decode_image(latents, vae_params):
        image = AutoencoderKL.from_config(f"{model_id}/vae").apply(
            {'params': vae_params}, 1 / 0.18215 * latents, method=AutoencoderKL.from_config(f"{model_id}/vae").decode
        ).sample
        image = (image / 2 + 0.5).clip(0, 1)
        return image.transpose(0, 2, 3, 1) # to HWC

    # 4. Denoising and Guidance Loop
    print("Starting denoising and vector guidance loop...")
    prng_seed = jax.random.PRNGKey(seed)
    
    latents_shape = (num_devices, 4, image_height // 8, image_width // 8)
    latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=dtype)
    
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    print("Preparing text embeddings...")
    text_embeddings = pmap_prepare_text_embeddings(prompt_ids, uncond_prompt_ids, params['text_encoder'])
    
    svg_params_guidance = {
        'num_colors': 6, 'simplification_epsilon_factor': 0.02,
        'min_contour_area': (guidance_resolution / 512)**2 * 30.0, 'max_features': 64
    }
    
    for i, t in enumerate(tqdm(timesteps)):
        
        noise_pred_both = pmap_predict_noise(latents, text_embeddings, t, params['unet'])
        noise_pred_uncond, noise_pred_text = jnp.split(noise_pred_both, 2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if guidance_start_step <= i < guidance_end_step and i % guidance_interval == 0:
            print(f"\nStep {i}: Performing vector guidance...")
            pred_original_sample = pmap_get_pred_original_sample(noise_pred_cfg, t, latents)
            decoded_image_tpu = pmap_decode_image(pred_original_sample, params['vae'])
            
            image_on_cpu = jax.device_get(decoded_image_tpu[0]) # Sync device 0 to host
            
            if guidance_resolution < image_height:
                image_on_cpu_resized = cv2.resize(image_on_cpu, (guidance_resolution, guidance_resolution), interpolation=cv2.INTER_AREA)
            else:
                image_on_cpu_resized = image_on_cpu

            rendered_svg_cpu, _ = host_vectorize_and_render(image_on_cpu_resized, **svg_params_guidance)
            
            # Since rendered_svg_cpu may be smaller, resize it back for loss calculation
            if rendered_svg_cpu.shape != image_on_cpu.shape:
                 rendered_svg_cpu_resized = cv2.resize(rendered_svg_cpu, (image_on_cpu.shape[1], image_on_cpu.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                 rendered_svg_cpu_resized = rendered_svg_cpu

            loss = np.mean((image_on_cpu - rendered_svg_cpu_resized) ** 2)
            grad_adjustment = (noise_pred_text - noise_pred_uncond) * loss * vector_guidance_scale
            noise_pred_cfg = noise_pred_cfg + grad_adjustment
            print(f"Guidance loss: {loss:.4f}, adjustment applied.")

        latents = pmap_scheduler_step(noise_pred_cfg, t, latents)
        if i % 10 == 0: gc.collect()

    # 5. Final Image Generation
    print("Generating final image and SVG...")
    final_image_tpu = pmap_decode_image(latents, params['vae'])
    final_image_cpu = jax.device_get(final_image_tpu[0])
    
    pil_image = Image.fromarray((final_image_cpu * 255).astype(np.uint8))
    raster_path = os.path.splitext(output_path)[0] + ".png"
    pil_image.save(raster_path)
    print(f"Saved final raster image to: {raster_path}")

    final_svg_params = {
        'num_colors': 24, 'simplification_epsilon_factor': 0.002,
        'min_contour_area': 1.0, 'max_features': 0
    }
    _, final_svg_string = host_vectorize_and_render(final_image_cpu, **final_svg_params)
    
    with open(output_path, "w") as f:
        f.write(final_svg_string)
    print(f"Saved final SVG to: {output_path}")

    return pil_image, final_svg_string

if __name__ == '__main__':
    PROMPT = "vector illustration of a gray wool coat with a faux fur collar, flat design, solid colors, minimalist, clean lines"
    NEGATIVE_PROMPT = "photo, realistic, 3d, noisy, texture, blurry, shadow, gradient, complex details"

    NUM_STEPS = 27
    VECTOR_GUIDANCE_SCALE = 2.0
    GUIDANCE_START = int(NUM_STEPS * 0.1)
    GUIDANCE_END = int(NUM_STEPS * 0.8)
    
    # Import necessary model classes for pmap functions
    from diffusers.models import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextConfig, FlaxCLIPTextModel
    
    # Run the generation
    generate_svg_with_guidance_tpu(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=7.5,
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=256,
        guidance_interval=3,
        output_path="out_tpu_optimized.svg",
        seed=42,
    )
