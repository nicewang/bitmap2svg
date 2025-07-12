import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# # ===== 1 =====

# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# # model_id = "/kaggle/input/stable-diffusion-v2/pytorch/1/1"
# model_id = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# ===== 2 =====

import torch

# Ensure GPU is being used and optimize for speed
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler

import kagglehub

import bitmap2svg

# Load with optimized scheduler and half precision
stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")

# DDIMScheduler
scheduler = DDIMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")
# # DPMSolverMultistepScheduler
# scheduler = DPMSolverMultistepScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_path,
    scheduler=scheduler,
    torch_dtype=torch.float16,  # Use half precision
    safety_checker=None         # Disable safety checker for speed
)

# Move to GPU and apply optimizations
pipe.to(device) 

def generate_bitmap(prompt, negative_prompt="", num_inference_steps=20, guidance_scale=15, generator=torch.Generator(device=device).manual_seed(42)):
    with torch.no_grad(): 
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
    torch.cuda.empty_cache()
    
    return image

def gen_bitmap(description):
    # prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
    prompt = f'{description}'
    bitmap = generate_bitmap(prompt, self.negative_prompt, self.num_inference_steps, self.guidance_scale)
    return bitmap

def predict(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float
) -> str:
    
    bitmap = generate_bitmap(prompt, negative_prompt, num_inference_steps, guidance_scale)
    svg = bitmap2svg.bitmap_to_svg(bitmap, num_colors=8)

    return bitmap, svg

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

img, svg = predict(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=20
)
