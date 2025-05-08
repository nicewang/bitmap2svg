import kagglehub


import torch

from diffusers import StableDiffusionPipeline,  DDIMScheduler, DPMSolverMultistepScheduler

from PIL import Image

import Bitmap2SVGConverter
import DiffusionModelConfig

DIFFUSION_MODEL_CONFIG = DiffusionModelConfig()

class StableDiffusionModel:

    def __init__(self):

        # ===== Step 1: Environment Setup =====
        # Ensure GPU is being used and optimize for speed
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")   

        # ===== Step 2: Load Pretained Diffusion Model =====
        # Load with optimized scheduler and half precision
        stable_diffusion_path = kagglehub.model_download(DIFFUSION_MODEL_CONFIG.stable_diffusion_config)

        # ===== Step 3: Build Pipeline for Image Generation =====
        # # DDIMScheduler
        # scheduler = DDIMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")
        # DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            stable_diffusion_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,  # Use half precision
            safety_checker=None         # Disable safety checker for speed
        )

        # Move to GPU and apply optimizations
        self.pipe.to(device) 

        # ===== Step 4: Set Initial Config =====
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.prompt_prefix = DIFFUSION_MODEL_CONFIG.prompt_prefix
        self.prompt_suffix = DIFFUSION_MODEL_CONFIG.prompt_suffix
        self.negative_prompt = DIFFUSION_MODEL_CONFIG.negative_prompt
        self.num_inference_steps = int(DIFFUSION_MODEL_CONFIG.num_inference_steps)
        self.guidance_scale = int(DIFFUSION_MODEL_CONFIG.guidance_scale)
        # self.num_attempt = 3

    def generate_bitmap(self, description: str) -> Image: 
        prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.guidance_scale,
        ).images[0]
        
        return image

    def predict_impl(self, prompt: str) -> str:
        bitmap = self.generate_bitmap(prompt)
        svg = Bitmap2SVGConverter.bitmap_to_svg_layered(bitmap)
        
        if svg is None:
            svg = self.default_svg

        return svg, bitmap

    def predict(self, prompt: str) -> str:
        # svg, img = self.predict_impl(prompt), but we don't need the bitmap image
        svg, _ = self.predict_impl(prompt)
        return svg
    
    def set_config(self, config: dict):
        if "prompt_prefix" in config:
            self.prompt_prefix = config["prompt_prefix"]
        if "prompt_suffix" in config:
            self.prompt_suffix = config["prompt_suffix"]
        if "negative_prompt" in config:
            self.negative_prompt = config["negative_prompt"]
        if "num_inference_steps" in config:
            self.num_inference_steps = int(config["num_inference_steps"])
        if "guidance_scale" in config:
            self.guidance_scale = int(config["guidance_scale"])
