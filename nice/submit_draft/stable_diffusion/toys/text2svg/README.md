### 1. Design Ideas

### 2. Versions
[1]

[
	2, 2_opt, 2_opt_2, [
			2_opt_3, 2_opt_3_perceploss_vgg, [
			2_opt_3_lpips, 2_opt_3_lpips_2, [
                2_opt_3_lpips_clip, [
                    2_opt_3_lpips_clip_textimg,[
                        2_opt_3_lpips_clip_textimg_progressive_guidance
                    ]
                ]
            ], [
			    2_opt_3_lpips_coopt, 2_opt_3_lpips_coopt_differentiable
			]
		]
	], 2_tpu
]

- cuda out of memory: 2,2_opt,2_opt_2
- runable: 2_opt_3
- 2_opt_3_perceploss_vgg: not useful
- 2_opt_3_lpips:
    - Increasing lambda (e.g. 0.2, 0.5): will make the model focus more on accurate color and brightness matching, possibly sacrificing some structural flexibility.
    - Decreasing lambda (e.g. 0.05, 0.01): will allow the model to more freely adjust the colors to better match the texture and shape, but may cause the final SVG to have a different tone than the original image.
- text2svg_2_opt_3_lpips_2.ipynb is the second running of 2_opt_3_lpips (not 2_opt_3_lpips_2)
- 2_opt_3_lpips_clip:
	- $$L_{total} = L_{reconstruction} + \lambda_{clip} \cdot L_{clip}$$

### Papers
* Directly Optimize Vector Params (Update Vector Params)
    * [CLIPasso](../../../../paper/CLIPasso/)
    * [VectorFusion](../../../../paper/VectorFusion/)
    * [SVGDreamer](../../../../paper/SVGDreamer/)
    * [livesketch](../../../../paper/livesketch/)
    * [DiffSketcher](../../../../paper/DiffSketcher/)
* Dependencies:
    * [DiffVG](../../../../paper/DiffVG/)
    * [Bézier Splatting for Fast and Differentiable Vector Graphics Rendering](../../../../paper/bezier_splatting_for_fast_and_differentiable_vector_graphics_rendering/)

### Supplement Materials
| Feature | Traditional Bitmap to SVG Algorithms (e.g., vtracer) | Differentiable SVG Generation |
| :--------------- | :------------------------------------------------------- | :-------------------------------------------------- |
| **Working Principle** | Rule-based heuristics using pixel analysis, edge detection, curve fitting | Parameterized SVG optimized via gradient descent using a differentiable renderer |
| **Core Mechanism** | Rules and heuristics | Gradient descent and loss function minimization |
| **Differentiability** | **No** (Black box, non-differentiable) | **Yes** (End-to-end differentiable) |
| **Semantic Understanding** | Weak (pixel-level features only) | Strong (integrates pre-trained perceptual models) |
| **Creativity** | None (replication only) | High (can achieve text-to-vector, style transfer, etc.) |
| **Output File Quality** | Potentially redundant, fragmented, difficult to edit | Cleaner, more semantic, easier to edit (theoretically) |
| **Computational Efficiency** | High (for simple images) | Low (requires rendering and backpropagation in each iteration) |
| **Training Required** | No | Typically **no large-scale model training from scratch**, but requires an optimization process. Some methods might train small, parameterized generators. |
| **Implementation Complexity** | Relatively simple | Complex, involves deep learning and graphics knowledge |
| **Typical Applications** | Logo vectorization, scanned drawing vectorization, image contour extraction | AI drawing (direct SVG output), vector style transfer, text-to-icon generation |