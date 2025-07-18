### Exp-1
Param:
```Python
# No Progressive Guidance

prompt = (
    f"clean classic vector illustration of {description}, "
    "flat design, solid colors only, minimalist, simple shapes, "
    "geometric style, flat color blocks, minimal details, no complex details"
)

# "lines, framing, hatching, background, patterns, outlines, "
negative_prompt = (
    "photo, realistic, 3d, noisy, textures, blurry, shadow, "
    "gradient, complex details, patterns, stripes, dots, "
    "repetitive elements, small details, intricate designs, "
    "busy composition, cluttered"
)

seed = 42
device = "cuda:0"

img, svg = generate_svg_with_guidance(
    prompt=prompt,
    negative_prompt=negative_prompt,
    description=description,
    device=device,
    # --- Strength parameter for blending structured and random noise ---
    strength=1.0, # 1.0 is pure noise, 0.0 is pure structure
    num_inference_steps=15,
    guidance_scale=20,
    vector_guidance_scale=4.5,
    # ToDo: parameter adjustment
    lpips_mse_lambda=0.1, 
    clip_guidance_scale=0.0, 
    # ToDo-End: parameter adjustment
    guidance_start_step=0,
    guidance_end_step=15,
    guidance_resolution=1024,
    guidance_interval=1,
    seed=42,
    use_half_precision=True,
    enable_sequential_cpu_offload=True,
    low_vram_shift_to_cpu=False
)
```
```
clip_guidance_scale: 0.0
score: 0.5389758799109624
clip_guidance_scale: 0.25
score: 0.5076237391672697
clip_guidance_scale: 0.50
score: 0.5118086584020818
clip_guidance_scale: 0.75
score: 0.5237646509131556
clip_guidance_scale: 0.80
score: 0.5261173600778374
clip_guidance_scale: 0.85
score: 0.5271939252389335
clip_guidance_scale: 0.90
score: 0.5266083447264027
clip_guidance_scale: 0.925
score: 0.5260514216904597
clip_guidance_scale: 0.95
score: 0.5304227564028253
clip_guidance_scale: 0.97
score: 0.49924003005101464
clip_guidance_scale: 0.99
score: 0.516175245000283
```
```Python
import matplotlib.pyplot as plt

x = [0.0, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.97, 0.99]
y = [0.5389758799109624, 0.5076237391672697, 0.5118086584020818, 0.5237646509131556, 0.5261173600778374, 0.5271939252389335, 0.5266083447264027, 0.5260514216904597, 0.5304227564028253, 0.49924003005101464, 0.516175245000283]

plt.plot(x, y)

plt.title("Scores among Different Clip Guidance")
plt.xlabel("clip_guidance_scale")
plt.ylabel("score")

plt.show()
```
![exp-1](exp_1.png)