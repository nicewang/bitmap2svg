import kagglehub
from latent_to_svg import CustomStableDiffusionPipeline

stable_diffusion_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")


text_to_svg_pipeline = CustomStableDiffusionPipeline(model_id=stable_diffusion_path)
prompt = "a beautiful landscape with mountains and trees"

print("Generating SVG sequence from text prompt...")
svg_codes = text_to_svg_pipeline.text_to_svg(
    prompt=prompt,
    num_inference_steps=27,
    guidance_scale=20,
    capture_every_n_steps=10
)

svg_code_list = []
for i, svg_code in enumerate(svg_codes):
    svg_code_list.append(svg_code)
    
final_image = text_to_svg_pipeline.generate_final_image(prompt)
