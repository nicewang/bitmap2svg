class DiffusionModelConfig:
    
    @property
    def stable_diffusion_config(self):
        return "stabilityai/stable-diffusion-v2/pytorch/1/1"
    
    @property
    def prompt_prefix(self):
        return "Simple, classic image of"
    
    @property
    def prompt_suffix(self):
        return "with flat color blocks, beautiful, minimal details, solid colors only"
    
    @property
    def negative_prompt(self):
        return "lines, framing, hatching, background, textures, patterns, details, outlines"
    
    @property
    def num_inference_steps(self):
        return 25
    
    @property
    def guidance_scale(self):
        return 20