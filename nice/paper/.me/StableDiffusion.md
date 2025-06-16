## Level-1:

**Keep in Mind**: Check whether - Did **Indeed Work** on **real-world problems**。

See `5` and `6`

#### 1. What is the problem?
Expensive taining (due to "operate directly in pixel space") and inference (due to sequential evaluations) consumptions of DMs.

#### 2. Why need to solve this problem?
**Computational Efficiency**: Enable DM training on limited computational resources.

#### 3. How is it different from prev.?
Apply DMs in **latent space** of ...

* An **autoencoder** for **low-diminutional latent space** representation learning - Reconstruction
	* A *Perceptual Equivalent* of prev.'s *perceptual compression* stage

* DM learn in latent space - Generation
	* Better *Scaling Properties* compared to those learn from *high-dimensional pixel space*

#### 4. Why is it better than prev.? (Advantages)
- **More Efficient**: It allows to achieve trade-off(? near-optimal point) between complexity reduction and detail preservation.
	-  Complexity Reduction: 
		-  *Learned Latent Space* is *"Perceptual Equivalent"* of *Perceptual Compression* without *Excessive Downsampling*
		-  Generation inputs at *Low-Dimensional Latent Space* rather than *High-Dimensional Pixel Space*
	-  Detail Preservation: Using *Autoencoder* to generate latent space representation is *Perceptual Equivalent* of *Perceptual Compression*
- **Retain Performance**: Enables general conditioning inputs (text & bounding boxes, etc) and high-resolution sythesis.

	- Better *Scaling Properties* (generator learn in latent space) compared to those learn from *high-dimensional pixel space*
	
	- Compared with *training an encoder/decoder architecture and a score-based prior simultaneously* (**one stage**):	
		- More stable training process: *No delicate weighting* of *reconstruction* and *generative* abilities
		- "**2-Stage** Separate Training" (Reconstruction+Generation): The autoencoder can focus on the reconstruction task without being disturbed by the generation task.
		- *Very little regularization* of the latent space

- On Diverse Tasks and More Flexible (also the supplements to **Retain Performance**):

	- For **Densely Conditioned** Tasks (super-resolution, inpainting, semantic synthesis, etc.): Could be applied in a *convolutional fashion* and render large

	- For **Class-Conditional** Tasks (text-to-image, layout-to-image, etc.): A *general-purpose conditioning mechanism* based on *cross-attention*, enabling *multi-modal* training  -> (conditioning inputs (text & bounding boxes, etc))

	- **Unconditional** Image Generation Tasks

| | Densely Conditioned | Class-Conditional |
|-----------|-------------------|-------------------|
| **Definition** | **Pixel-level Correspondence**: The conditional information (input) and the output image have a close pixel-level correspondence in space. | **Global Semantics**: Conditional information provides global semantics or style guidance. |
| | **Structured Conditions**: The conditional input is usually structured data of the same size (or pixel-level magnification) as the output. | **Abstract Conditions**: Conditions are usually abstract information such as text, labels or layout. |
| | **Strong Local Constraints**: The generation of each pixel position is strongly constrained by the corresponding conditions | **More Creativity**: Greater freedom of generation under the constraints of conditions. |
| **Condition Density** | Pixel-level dense constraints | Global sparse constraints |
| **Spatial Correspondence** | Strong spatial correspondence | Weak spatial correspondence |
| **Generation Freedom** | Lower (fixed structure) | Higher (more creative) |
| **Task Difficulty** | Mainly detail reconstruction | Mainly semantic understanding |
| **Evaluation Method** | Pixel-level metrics dominant | Semantic consistency dominant |
| **Applications** | **Super-Resolution**: low-resolution image → high-resolution image (pixel-level magnification) | **Text-to-Image** (semantic guidance) |
|  | **Inpainting**: Image with mask → Complete image corresponded spatial position of input and output) | **Layout-to-Image**: layout box + category label → image (layout guidance) |
|  | **Semantic Synthesis**: semantic segmentation map → real image (corresponding to each pixel category) |  |

#### 5. What is the approach itself?
**Latent** DMs with **cross-attention** layers.

- Latent DMs: Autoencoder *Reconstructor* + DM *Generator*
- *Conditioning Mechanisms* based on *cross-attention*

#### 6. What are the applications of it?

**Efficient**:

- class-conditional image synthesis
- unconditional image generation
- image inpainting
- super-resolution
- etc.

#### Comparison of Image Synthesis Models

| Model Type | Main Features | Pros | Cons | Best For |
|------------|---------------|------|------|----------|
| **AR Transformers** | - Billions of parameters <br> - Sequential generation | High-quality complex scenes | Extremely expensive computation | High-resolution synthesis |
| **GANs** | - Adversarial training <br> - Fast generation | Fast, sharp results | - Training instability <br> - Mode collapse <br> - Hard to modeling complex, multi-modal distributions | - Simple datasets (limited variability) <br> - Face generation |
| **Diffusion Models** | - Sequential denoising autoencoders <br> - Parameter sharing | - Stable training <br> - Versatile <br> - SOTA results | Slower than GANs | Most image synthesis tasks |
