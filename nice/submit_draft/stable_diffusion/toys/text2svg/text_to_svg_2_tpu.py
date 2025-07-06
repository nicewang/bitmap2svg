import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P

import numpy as np
import cv2  # OpenCV for vectorization logic
import cairosvg # For rendering SVG on CPU

from PIL import Image
from tqdm.auto import tqdm
import os
import random
import gc

from diffusers import FlaxStableDiffusionPipeline
from flax.jax_utils import replicate
from flax.training.common_utils import shard

# Ensure JAX is using TPU
assert jax.default_backend() == 'tpu', 'This code is designed for TPU execution.'
num_devices = jax.device_count()
print(f"JAX detected {num_devices} TPU devices.")

# -----------------------------------------------------------------------------------
# Part 1: Host-side (CPU) Vectorization and Rendering Logic
# Re-implementation of the C++ bitmap2svg logic using Python/OpenCV
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
    for color_bgr in palette:
        # Create a mask for the current color
        mask_color = cv2.inRange(centers[labels.flatten()].reshape(img_rgb.shape), np.array(color_bgr), np.array(color_bgr))
        
        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert color BGR to HEX for SVG
        hex_color = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            # 4. Polygon Simplification
            epsilon = simplification_epsilon_factor * cv2.arcLength(contour, True)
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_poly) >= 3:
                all_polygons.append({
                    "points": approx_poly.reshape(-1, 2),
                    "color": hex_color,
                    "area": cv2.contourArea(contour)
                })

    # Sort polygons by area (descending) to draw larger shapes first
    all_polygons.sort(key=lambda p: p["area"], reverse=True)
    if max_features > 0:
        all_polygons = all_polygons[:max_features]

    # 5. Build SVG string
    # Use average color for the background
    avg_color_bgr = np.mean(pixels, axis=0).astype(int)
    bg_hex_color = f"#{avg_color_bgr[2]:02x}{avg_color_bgr[1]:02x}{avg_color_bgr[0]:02x}"

    svg_elements = [f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>']
    for poly in all_polygons:
        points_str = " ".join([f"{p[0]},{p[1]}" for p in poly["points"]])
        svg_elements.append(f'<polygon points="{points_str}" fill="{poly["color"]}"/>')
    
    svg_string = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(svg_elements)}</svg>'

    # 6. Render SVG back to a bitmap using CairoSVG
    try:
        # CairoSVG expects bytes
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        
        # Decode PNG data to an image
        rendered_img_np = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_COLOR)
        rendered_img_np = cv2.cvtColor(rendered_img_np, cv2.COLOR_BGR2RGB) # back to RGB
        rendered_img_np = rendered_img_np.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Warning: SVG rendering failed: {e}. Returning a black image.")
        rendered_img_np = np.zeros_like(image_np, dtype=np.float32)

    return rendered_img_np, svg_string

# -----------------------------------------------------------------------------------
# Part 2: JAX/TPU Accelerated Generation Function
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
    """
    Main function for text-to-svg generation on TPU.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    print(f"Using prompt: '{prompt}'")
    print(f"Using seed: {seed}. Running on {num_devices} TPU devices.")

    # 1. Load Models and move them to JAX/Flax
    # This will download the PyTorch weights and convert them on the fly
    pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        model_id,
        revision="flax",
        dtype=dtype,
        # Safety checker can be removed if not needed, to save memory
        # safety_checker=None 
    )
    
    # 2. Define JAX PRNG key and replicate parameters across devices
    prng_seed = jax.random.PRNGKey(seed)
    params = replicate(params)
    
    # 3. Process prompts
    prompts = [prompt] * num_devices
    neg_prompts = [negative_prompt] * num_devices
    
    prompt_ids = pipeline.prepare_inputs(prompts)
    uncond_prompt_ids = pipeline.prepare_inputs(neg_prompts)

    # Shard inputs for pmap
    prompt_ids = shard(prompt_ids)
    uncond_prompt_ids = shard(uncond_prompt_ids)

    # 4. Prepare latents
    latents_shape = (
        num_devices,
        pipeline.unet.config.in_channels,
        pipeline.height // pipeline.vae_scale,
        pipeline.width // pipeline.vae_scale
    )
    latents = jax.random.normal(prng_seed, shape=latents_shape, dtype=dtype)
    
    # 5. Define the core TPU computation steps using `pmap`
    # `pmap` compiles the function and runs it in parallel on all available devices.
    
    # This step predicts the noise using UNet
    @jax.pmap
    def pmap_predict_noise(latents, text_embeddings, t):
        # expand the latents if we are doing classifier-free guidance
        latent_model_input = jnp.concatenate([latents] * 2)
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipeline.unet.apply(
            {'params': params['unet']},
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        return noise_pred

    # This step updates the latents using the scheduler
    @jax.pmap
    def pmap_scheduler_step(noise_pred, t, latents):
        return pipeline.scheduler.step(noise_pred, t, latents).prev_sample

    # This step decodes latents into an image
    @jax.pmap
    def pmap_decode_image(latents):
        image = pipeline.vae.apply(
            {'params': params['vae']},
            1 / pipeline.vae.config.scaling_factor * latents,
            method=pipeline.vae.decode
        ).sample
        image = (image / 2 + 0.5).clip(0, 1)
        # Convert to HWC for standard image processing
        image = image.transpose(0, 2, 3, 1)
        return image

    # 6. Denoising and Guidance Loop
    print("Starting denoising and vector guidance loop...")
    pipeline.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipeline.scheduler.timesteps
    
    # Get text embeddings once, outside the loop
    text_embeddings = pipeline.prepare_text_embeddings(prompt_ids, uncond_prompt_ids, params)
    
    # Parameters for the CPU-side vectorizer
    svg_params_guidance = {
        'num_colors': 6,
        'simplification_epsilon_factor': 0.02,
        'min_contour_area': (guidance_resolution / 512)**2 * 30.0,
        'max_features': 64
    }
    
    for i, t in enumerate(tqdm(timesteps)):
        
        # --- TPU Computation ---
        noise_pred = pmap_predict_noise(latents, text_embeddings, t)
        
        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2)
        noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # --- Vector Guidance (Hybrid CPU/TPU) ---
        if guidance_start_step <= i < guidance_end_step and i % guidance_interval == 0:
            print(f"\nStep {i}: Performing vector guidance...")
            
            # a. Predict original sample based on current noise prediction
            pred_original_sample = pipeline.scheduler.step(noise_pred_cfg, t, latents).pred_original_sample
            
            # b. Decode the image on TPU
            decoded_image_tpu = pmap_decode_image(pred_original_sample)
            
            # c. Transfer ONE image from TPU device 0 to Host CPU
            # jax.device_get will sync the device and copy data to host RAM
            image_on_cpu = jax.device_get(decoded_image_tpu[0])
            
            # d. Resize if needed
            if guidance_resolution < pipeline.height:
                image_on_cpu = cv2.resize(image_on_cpu, (guidance_resolution, guidance_resolution), interpolation=cv2.INTER_AREA)
            
            # e. Run vectorizer on CPU
            rendered_svg_cpu, _ = host_vectorize_and_render(image_on_cpu, **svg_params_guidance)
            
            # f. Calculate loss on CPU and create guidance gradient
            loss = np.mean((image_on_cpu - rendered_svg_cpu) ** 2)
            grad_adjustment = (noise_pred_text - noise_pred_uncond) * loss * vector_guidance_scale
            
            # g. Apply the adjustment back on the TPU
            noise_pred_cfg = noise_pred_cfg + grad_adjustment
            print(f"Guidance loss: {loss:.4f}, adjustment applied.")

        # --- Update latents on TPU ---
        latents = pmap_scheduler_step(noise_pred_cfg, t, latents)

        # Cleanup host memory periodically
        if i % 10 == 0:
            gc.collect()

    # 7. Final Image Generation
    print("Generating final image and SVG...")
    
    # Decode final latents on TPU and get one image from device 0
    final_image_tpu = pmap_decode_image(latents)
    final_image_cpu = jax.device_get(final_image_tpu[0])
    
    # Save raster image
    pil_image = Image.fromarray((final_image_cpu * 255).astype(np.uint8))
    raster_path = os.path.splitext(output_path)[0] + ".png"
    pil_image.save(raster_path)
    print(f"Saved final raster image to: {raster_path}")

    # Create final high-quality SVG on CPU
    final_svg_params = {
        'num_colors': 24,
        'simplification_epsilon_factor': 0.002,
        'min_contour_area': 1.0,
        'max_features': 0 # Unlimited
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
