import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from typing import List, Optional, Union

import logging

class LatentToSVG:
    def __init__(self, latent_dim=4*64*64, device=None):
        self.latent_dim = latent_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.svg_generator = self.build_svg_network()
        self.svg_generator.to(self.device).half()
    
    def build_svg_network(self):
        """Build a neural network for converting latent to SVG parameters"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 200),  # Output SVG path parameters (e.g. 20 path points * 10 parameters)
        )
    
    def to(self, device):
        """Move model to corresponding device"""
        self.device = device
        self.svg_generator.to(device)
        return self
    
    def latent_to_svg_params(self, latent):
        """Covert latent to SVG path parameters"""
        # Make sure the latent on the right device
        if latent.device != self.device:
            latent = latent.to(self.device)
        
        if len(latent.shape) == 4:  # [batch, channels, height, width]
            latent_flat = latent.flatten(start_dim=1)
        else:
            latent_flat = latent.flatten()
        
        # Make sure the network is in eval mode
        self.svg_generator.eval()
        with torch.no_grad():
            svg_params = self.svg_generator(latent_flat)
        return svg_params
    
    def params_to_svg(self, params, width=512, height=512):
        """Convert parameters to SVG code"""
        # Reshape parameters into waypoints
        params = params.cpu().numpy() if torch.is_tensor(params) else params
        
        # Normalize parameters to image size
        coords = params.reshape(-1, 2)
        coords[:, 0] = (coords[:, 0] + 1) * width / 2  # x-axis
        coords[:, 1] = (coords[:, 1] + 1) * height / 2  # y-axis
        
        # Build SVG Path
        path_data = f"M {coords[0, 0]:.2f} {coords[0, 1]:.2f}"
        for i in range(1, len(coords)):
            path_data += f" L {coords[i, 0]:.2f} {coords[i, 1]:.2f}"
        
        svg_code = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <path d="{path_data}" fill="none" stroke="black" stroke-width="2"/>
</svg>"""
        return svg_code

class CustomStableDiffusionPipeline:
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)
        
        # Initialize the SVG generator and move to the correct device
        self.svg_generator = LatentToSVG(device=self.device)
        self.intermediate_latents = []
    
    def custom_scheduler_step(self, scheduler, model_output, timestep, sample):
        """Custom scheduler step to capture intermediate latent vector"""
        # Store intermediate latent
        self.intermediate_latents.append(sample.clone())
        
        # Run the normal scheduler steps
        return scheduler.step(model_output, timestep, sample).prev_sample
    
    def text_to_svg(self, 
                   prompt: str, 
                   num_inference_steps: int = 27,
                   guidance_scale: float = 20,
                   height: int = 512,
                   width: int = 512,
                   capture_every_n_steps: int = 10) -> List[str]:
        """
        Generate SVG from text prompt
        
        Args:
            prompt: text description
            num_inference_steps: inference steps (more for better quality / slower)
            guidance_scale: guidance scale (how tightly to follow prompts)
            height: image height
            width: image width
            capture_every_n_steps: how many steps to capture a latent
        
        Returns:
            SVG code list
        """
        
        # Clear the former intermediate latent
        self.intermediate_latents = []
        
        # Encoded text prompt
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.pipe.device))[0]
        
        # Unconditional embedding (for classifier-free guidance)
        uncond_tokens = [""]
        uncond_inputs = self.pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            uncond_embeddings = self.pipe.text_encoder(uncond_inputs.input_ids.to(self.pipe.device))[0]
        
        # Merge conditions and unconditional embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latent
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            device=self.pipe.device,
            dtype=text_embeddings.dtype
        )
        
        # Set scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # Extend latent for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Noise prediction
            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Run classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Update latent and capture intermediate results
            if i % capture_every_n_steps == 0:
                self.intermediate_latents.append(latents.clone())
            
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Add final latents
        self.intermediate_latents.append(latents)
        
        # Convert intermediate latents to SVG
        svg_codes = []
        
        for i, latent in enumerate(self.intermediate_latents):
            try:
                #  Make sure the latent on the right device
                if latent.device != self.device:
                    latent = latent.to(self.device)
                
                svg_params = self.svg_generator.latent_to_svg_params(latent)
                
                # Take the first if batch processing
                if len(svg_params.shape) > 1:
                    svg_params = svg_params[0]
                
                svg_code = self.svg_generator.params_to_svg(svg_params, width, height)
                svg_codes.append(svg_code)
                
            except Exception as e:
                logging.error(f"Error converting latent {i}: {e}")
                continue
        
        return svg_codes
    
    def generate_final_image(self, prompt: str, **kwargs):
        """Generate the final image (standard Stable Diffusion output)"""
        return self.pipe(prompt, **kwargs).images[0]
