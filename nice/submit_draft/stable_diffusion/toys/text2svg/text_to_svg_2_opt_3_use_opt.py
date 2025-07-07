from text_to_svg_2_opt_3 import generate_svg_with_guidance

def gen_bitmap(self, description):
    """
    Generate bitmap and SVG from text description with optimized prompt management
    to avoid CLIP tokenizer sequence length limits (max 77 tokens).
    """
    
    # Pattern keywords that might generate repetitive elements
    pattern_keywords = [
        'checkered', 'striped', 'plaid', 'polka dot', 'mesh', 'grid',
        'fabric', 'textile', 'woven', 'knitted', 'corduroy', 'tweed',
        'houndstooth', 'argyle', 'paisley', 'floral pattern'
    ]
    
    PROMPT = (
        f"clean classic vector illustration of {description}, "
        "flat design, solid colors only, minimalist, simple shapes, "
        "geometric style, flat color blocks, minimal details, no complex details"
    )
    # self.prompt_suffix = "with flat color blocks, beautiful, minimal details, solid colors only"
    # self.negative_prompt = "lines, framing, hatching, background, textures, patterns, details, outlines"
    # Standard negative prompt (carefully controlled length)
    NEGATIVE_PROMPT = (
        "photo, realistic, 3d, noisy, textures, blurry, shadow, "
        "lines, framing, hatching, background, patterns, outlines, "
        "gradient, complex details, patterns, stripes, dots, "
        "repetitive elements, small details, intricate designs, "
        "busy composition, cluttered"
    )
    
    # Generation parameters
    NUM_STEPS = 27
    VECTOR_GUIDANCE_SCALE = 5.0  # Force simplification
    # GUIDANCE_START = int(NUM_STEPS * 0.05)  # Start guidance earlier
    # GUIDANCE_END = int(NUM_STEPS * 0.85)    # End guidance later
    GUIDANCE_START = 0
    GUIDANCE_END = NUM_STEPS
    
    # Call the generation function with optimized parameters
    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=25.0,  # Slightly increased CFG scale
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=256,
        guidance_interval=1,  # Reduce interval to increase guidance frequency
        output_path="out_optimized.svg",
        seed=42,
        enable_attention_slicing=True,
        enable_cpu_offload=True,
        use_half_precision=True,
        enable_sequential_cpu_offload=True,
        low_vram_shift_to_cpu=True
    )
    return svg, img

def gen_bitmap(self, description):
    """
    Generate bitmap and SVG from text description with optimized prompt management
    to avoid CLIP tokenizer sequence length limits (max 77 tokens).
    """
    
    # Pattern keywords that might generate repetitive elements
    pattern_keywords = [
        'checkered', 'striped', 'plaid', 'polka dot', 'mesh', 'grid',
        'fabric', 'textile', 'woven', 'knitted', 'corduroy', 'tweed',
        'houndstooth', 'argyle', 'paisley', 'floral pattern'
    ]
    
    # Clothing keywords for fashion-specific prompts
    clothing_keywords = ['pants', 'overalls', 'shirt', 'dress', 'jacket']
    
    # Check for pattern and clothing keywords
    has_pattern = any(keyword in description.lower() for keyword in pattern_keywords)
    has_clothing = any(keyword in description.lower() for keyword in clothing_keywords)
    
    # Generate optimized prompts based on content type
    if has_pattern:
        # For pattern-containing descriptions, force extreme simplification
        PROMPT = (
            f"simple vector icon of {description}, "
            "solid colors only, flat design, geometric abstraction, "
            "logo style, minimal details, no patterns, no textures"
        )
        
        # Specialized negative prompt for pattern content (avoid token limit)
        NEGATIVE_PROMPT = (
            "realistic textures, actual patterns, fabric details, "
            "complex surfaces, photo, 3d, gradient, shadow, "
            "intricate designs, busy composition, repetitive elements"
        )
        
    elif has_clothing:
        # For clothing items, use fashion illustration style
        PROMPT = (
            f"fashion illustration of {description}, "
            "flat lay style, solid colors, clean silhouette, "
            "minimalist fashion drawing, vector art, simple color blocks"
        )
        
        # Fashion-specific negative prompt
        NEGATIVE_PROMPT = (
            "photo, realistic, 3d, fabric texture, wrinkles, "
            "complex details, patterns, shadow, gradient, "
            "busy design, cluttered, intricate clothing details"
        )
        
    else:
        # Standard approach for regular descriptions
        PROMPT = (
            f"clean vector illustration of {description}, "
            "flat design, solid colors, minimalist, simple shapes, "
            "geometric style, bold colors, no complex details"
        )
        
        # Standard negative prompt (carefully controlled length)
        NEGATIVE_PROMPT = (
            "photo, realistic, 3d, noisy, texture, blurry, shadow, "
            "gradient, complex details, patterns, stripes, dots, "
            "repetitive elements, small details, intricate designs, "
            "busy composition, cluttered"
        )
    
    # Generation parameters
    NUM_STEPS = 27
    VECTOR_GUIDANCE_SCALE = 2.5  # Force simplification
    GUIDANCE_START = int(NUM_STEPS * 0.05)  # Start guidance earlier
    GUIDANCE_END = int(NUM_STEPS * 0.85)    # End guidance later
    
    # Call the generation function with optimized parameters
    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=8.0,  # Slightly increased CFG scale
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=256,
        guidance_interval=2,  # Reduce interval to increase guidance frequency
        output_path="out_optimized.svg",
        seed=42,
        enable_attention_slicing=True,
        enable_cpu_offload=True,
        use_half_precision=True,
        enable_sequential_cpu_offload=True,
        low_vram_shift_to_cpu=True
    )
    
    return svg, img

# Additional utility function: Intelligently adjust prompt based on description
def create_adaptive_prompt(description):
    """Dynamically create the most suitable prompt based on description content"""
    
    # Define problematic keywords and corresponding handling strategies
    problematic_patterns = {
        'checkered': 'solid colored',
        'striped': 'plain',
        'plaid': 'solid colored',
        'polka dot': 'solid colored',
        'corduroy': 'smooth fabric',
        'tweed': 'solid fabric',
        'mesh': 'solid material',
        'grid': 'plain surface'
    }
    
    # Replace problematic description words
    cleaned_description = description.lower()
    for problematic, replacement in problematic_patterns.items():
        if problematic in cleaned_description:
            cleaned_description = cleaned_description.replace(problematic, replacement)
    
    # Build prompt
    base_prompt = f"vector illustration of {cleaned_description}"
    
    # Add specific style guidance based on content
    if any(word in description.lower() for word in ['pants', 'clothing', 'garment']):
        style_prompt = "fashion flat, solid colors, clean silhouette"
    elif any(word in description.lower() for word in ['geometric', 'abstract']):
        style_prompt = "simple geometric shapes, bold colors"
    else:
        style_prompt = "flat design, minimalist, clean lines"
    
    final_prompt = f"{base_prompt}, {style_prompt}, no patterns, no textures"
    
    return final_prompt

# Usage example
def gen_bitmap_adaptive(self, description):
    """Generate method using adaptive prompt"""
    
    PROMPT = create_adaptive_prompt(description)
    
    # Universal enhanced negative prompt
    NEGATIVE_PROMPT = (
        "photo, realistic, 3d, noisy, texture, blurry, shadow, gradient, "
        "complex details, patterns, stripes, dots, checkerboard, crosshatch, "
        "repetitive elements, small details, intricate designs, busy composition, "
        "cluttered, dense patterns, fabric texture, wood grain, mesh, grid patterns, "
        "stippling, fine lines, ornate details, realistic textures, "
        "detailed fabric, woven texture, knitted pattern, embroidery"
    )
    
    NUM_STEPS = 30  # Increase steps for better quality
    VECTOR_GUIDANCE_SCALE = 3.0  # Stronger vectorization guidance
    GUIDANCE_START = int(NUM_STEPS * 0.03)
    GUIDANCE_END = int(NUM_STEPS * 0.9)
    
    img, svg = generate_svg_with_guidance(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=8.5,
        vector_guidance_scale=VECTOR_GUIDANCE_SCALE,
        guidance_start_step=GUIDANCE_START,
        guidance_end_step=GUIDANCE_END,
        guidance_resolution=256,
        guidance_interval=2,
        output_path="out_optimized.svg",
        seed=42,
        enable_attention_slicing=True,
        enable_cpu_offload=True,
        use_half_precision=True,
        enable_sequential_cpu_offload=True,
        low_vram_shift_to_cpu=True
    )
    
    return svg, img
