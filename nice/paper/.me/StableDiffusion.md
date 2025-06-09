## Level-1:

**Keep in Mind**: Check whether - Did **Indeed Work** on **real-world problems**。

See 6

#### 1. What is the problem?
Expensive taining (due to "operate directly in pixel sapce") and inference (due to sequential evaluations) consumptions of DMs.

#### 2. Why need to solve this problem?
**Computational Efficiency**: Enable DM training on limited computational resources.

#### 3. How is it different from prev.?
Apply DMs in **latent space** of ...

#### 4. Why is it better than prev.? (Advantages)
- It allows to achieve trade-off(? near-optimal point) between complexity reduction and detail preservation.
- Enables general conditioning inputs (text & bounding boxes, etc) and high-resolution sythesis.

#### 5. What is the approach itself?
**Latent** DMs with **cross-attention** layers.

#### 6. What are the applications of it?

**Efficient**:

- text-to-image synthesis <-----------|(2)
- class-conditional image synthesis <-|(1)
- unconditional image generation <-|(4)
- image inpainting <----------------|(3)
- super-resolution

# Comparison of Image Synthesis Models

| Model Type | Main Features | Pros | Cons | Best For |
|------------|---------------|------|------|----------|
| **AR Transformers** | - Billions of parameters <br> - Sequential generation | High-quality complex scenes | Extremely expensive computation | High-resolution synthesis |
| **GANs** | - Adversarial training <br> - Fast generation | Fast, sharp results | - Training instability <br> - Mode collapse <br> - Hard to modeling complex, multi-modal distributions | - Simple datasets (limited variability) <br> - Face generation |
| **Diffusion Models** | - Sequential denoising autoencoders <br> - Parameter sharing | - Stable training <br> - Versatile <br> - SOTA results | Slower than GANs | Most image synthesis tasks |
