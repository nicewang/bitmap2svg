## Level 1: The Main Idea

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
		* Q: Why "better scaling properties"?
		* A: When the image resolution increases from 512×512 to 1024×1024 (for example), the computational cost of LDMs increases relatively modestly and remains feasible (since generator learn in the **relatively low-dimensional** latent space -> **easier to learn** when resolution increasing).

#### 4. Why is it better than prev.? (Advantages)
- **More Efficient**: It allows to achieve trade-off(? near-optimal point) between complexity reduction and detail preservation.
	-  Complexity Reduction: 
		-  *Learned Latent Space* is *"Perceptual Equivalent"* of *Perceptual Compression* without *Excessive Downsampling*
		-  Generation inputs at *Low-Dimensional Latent Space* rather than *High-Dimensional Pixel Space*
	-  Detail Preservation: Using *Autoencoder* to generate latent space representation is *Perceptual Equivalent* of *Perceptual Compression* while *ignoring high-frequency, imperceptible details*
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

- unconditional image generation (generated directly from *noise*, usually as a sub-step in the image synthesis process)
- densely conditioned generation:
	- image inpainting
	- super-resolution
- etc.

#### Comparison of Image Synthesis Models

| Model Type | Main Features | Pros | Cons | Best For |
|------------|---------------|------|------|----------|
| **AR Transformers** | - Billions of parameters <br> - Sequential generation | High-quality complex scenes | Extremely expensive computation | High-resolution synthesis |
| **GANs** | - Adversarial training <br> - Fast generation | Fast, sharp results | - Training instability <br> - Mode collapse <br> - Hard to modeling complex, multi-modal distributions | - Simple datasets (limited variability) <br> - Face generation |
| **Diffusion Models** | - Sequential denoising autoencoders <br> - Parameter sharing | - Stable training <br> - Versatile <br> - SOTA results | Slower than GANs | Most image synthesis tasks |

## Level 2: The Structure
## Level 3: The Most Important Details
### 1. UNet and Cross-Attention Based Conditioning Mechanism
#### UNet

```
Unified UNet Block = {
    - Conv Layer: Conv2D
    - Norm Layer: BatchNorm/GroupNorm/LayerNorm/etc. 
    - Activation: ReLU/SiLU/GELU/etc.
    - Attention Mechanism: Self-Attention/Cross-Attention
    - Residual Connection
}

Encoder Unet Block (generally) = {
	- Conv Layer: Conv2D
   	- Norm Layer: BatchNorm/GroupNorm/LayerNorm/etc. 
   	- Activation: ReLU/SiLU/GELU/etc.
} i.e. Downsample Block

Decoder Unet Block (generally) = {
	- Transposed Conv2D/Interpolation
	- Conv Layer: Conv2D
} i.e. Upsample Block
```
#### Special Design of UNet in Stable Diffusion
```
ResNet Block:
Input → Conv → Norm → Activation → Conv → Norm → Add → Output
  ↓                                               ↑
  └─────────────── Skip Connection ───────────────┘

Attention Block = {
	- Attention Mechanism: Self-Attention/Cross-Attention
}
```
```
ResNet (2015)

Input x → [Conv → ReLU → Conv] → Add → Output
    ↓                             ↑
    └────── Identity Mapping ─────┘
    
- Purpose: To solve the gradient disappearance problem of deep networks
- element-wise addition
```
```
Skip Connection in UNet (2015)

Encoder Features → [Bottleneck] → Decoder → Concat → Output
       ↓                                      ↑
       └────────── Skip Connection ───────────┘

- Purpose: Maintain spatial detail information
- channel-wise concatenation
```
example of UNet:

```
Encoder:
  Resolution Level 1 (64×64): ResNet + ResNet + CrossAttn  # i=1
  Resolution Level 2 (32×32): ResNet + ResNet + CrossAttn  # i=2
  Resolution Level 3 (16×16): ResNet + ResNet + CrossAttn  # i=3

Bottleneck:
  Level 4 (8×8):   ResNet + CrossAttn + ResNet  # i=4

Decoder:
  Resolution Level 3 (16×16): ResNet + CrossAttn + ResNet  # i=3  
  Resolution Level 2 (32×32): ResNet + CrossAttn + ResNet  # i=2
  Resolution Level 1 (64×64): ResNet + CrossAttn + ResNet  # i=1
```
So, the whole architecture of UNet may seems like following:

```
Encoder                    Decoder
Input                      Output
  ↓                          ↑
Block1 ─────────────────→ Block1'
  ↓           skip           ↑
Block2 ─────────────────→ Block2'  
  ↓           conn           ↑
Block3 ─────────────────→ Block3'
  ↓                          ↑
  └─────── Bottleneck ───────┘

e.g.
Resolution Level 1: 256×256×64  ────────→ 256×256×64
Resolution Level 2: 128×128×128 ────────→ 128×128×128  
Resolution Level 3: 64×64×256   ────────→ 64×64×256
Resolution Level 4: 32×32×512   ────────→ 32×32×512
                        ↓                    ↑
                        Bottleneck: 16×16×1024
```
For Python Pseudocode

```Python
# Encoder Period - Store Feature
encoder_features = []
x = input_image
for encoder_block in encoder_blocks:
    x = encoder_block(x)
    encoder_features.append(x)  # stored for skip connection
    x = downsample(x)

# Bottleneck
x = bottleneck(x)

# Decoder Period - Use skip connections
for i, decoder_block in enumerate(decoder_blocks):
    x = upsample(x)
    skip_feat = encoder_features[-(i+1)]  # Encoder Feature of corresponding level
    x = concat([x, skip_feat], dim=channel)  # concat
    x = decoder_block(x) # for φ_N-i-1, N is the total-cnt of resolution levels

```