[Version 17, Version 23, Version 26, Version 27, Version 28, Version 32, Version 33, Version 34, Version 36, Version 37, Version 38, Version 39, Version 40]

[Version 14, Version 19]

<font color=red>simplification\_epsilon\_factor=0.015->0.005: Very obvious **effect attenuation**</font>

```
Version 17:
num_inference_steps = 25
num_attempt = 3
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=45.0, num_colors=8)
Original bitmap score: 0.41999372640056415
Final svg score: 0.6954702749231211

Version 23:
num_inference_steps = 30
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=45.0, num_colors=8)
Original bitmap score: 0.44566469479250975
Final svg score: 0.6392523522905852

Version 26:
num_inference_steps = 45
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=15.0, num_colors=8)
Original bitmap score: 0.44129766133958587
Final svg score: 0.6610786231381424

Version 27:
num_inference_steps = 35
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.33338598137235526
Final svg score: 0.6765057045624071

Version 28:
num_inference_steps = 32
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.40706078202781587
Final svg score: 0.6912730022532566

Version 32:
num_inference_steps = 32
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0)
Original bitmap score: 0.4304560549572242
Final svg score: 0.6533476373759909

Version 33:
num_inference_steps = 32
guidance_scale = 25
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.42393225772854554
Final svg score: 0.6625488071143946

Version 34:
num_inference_steps = 32
guidance_scale = 25
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=15.0, num_colors=8)
Original bitmap score: 0.36695800540244045
Final svg score: 0.6663431791946264

Version 36:
num_inference_steps = 32
guidance_scale = 20
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, simplification_epsilon_factor=0.010, min_contour_area=15.0, num_colors=8)
Original bitmap score: 0.3612905121507316
Final svg score: 0.6621436583083569

Version 37:
def get_aes_and_ocr_score(svg_content):
    image_processor = metric.ImageProcessor(image=metric.svg_to_png(svg_content), seed=52).apply()
    ...
    ocr_score = metric.vqa_evaluator.ocr(image_processor.image)
    ...
num_inference_steps = 32
guidance_scale = 20
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, simplification_epsilon_factor=0.010, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.45583545929567
Final svg score: 0.6509088551163473

Version 38:
def get_aes_and_ocr_score(svg_content):
    image_processor = metric.ImageProcessor(image=metric.svg_to_png(svg_content), seed=52).apply()
    ...
    ocr_score = metric.vqa_evaluator.ocr(image_processor.image)
    ...
num_inference_steps = 32
guidance_scale = 20
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.43158260094816725
Final svg score: 0.656942386876851

Version 39:
num_inference_steps = 32
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, simplification_epsilon_factor=0.005, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.42513860841304313
Final svg score: 0.604263894918382

Version 40:
num_inference_steps = 32
num_attempt = 4
prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=20.0, num_colors=8)
Original bitmap score: 0.48676969365758704
Final svg score: 0.6921606004990434
```

[Version 15, Version 24, Version 25, Version 29] -> no generator

```
Version 15:
prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
num_inference_steps = 25
num_attempt = 3
svg = bitmap2svg.bitmap_to_svg(bitmap)
Original bitmap score: 0.470562931493742
Final svg score: 0.6734555091769231

Version 24:
prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
num_inference_steps = 30
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap)
Original bitmap score: 0.5532725086267603
Final svg score: 0.6555382395071472

Version 25:
prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
num_inference_steps = 35
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=15.0)
Original bitmap score: 0.5275079692029931
Final svg score: 0.6604855959343606

Version 29:
prompt = f'{description}'
num_inference_steps = 32
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=15.0)
Original bitmap score: 0.2706269242860204
Final svg score: 0.6720106723585885

Version 41:
prompt = f'{self.prompt_prefix} {description} {self.prompt_suffix}'
num_inference_steps = 35
num_attempt = 4
svg = bitmap2svg.bitmap_to_svg(bitmap, min_contour_area=15.0, num_colors=8)
Original bitmap score: 0.4362707876475086
Final svg score: 0.6624359019852176
```

[Version 22, Version 31]

```
Version 22:
base
Original bitmap score: 0.4363026800935096
Final svg score: 0.6550174730417202

Versio 31:
1.
DPMSolverMultistepScheduler->EulerAncestralDiscreteScheduler
2.
num_inference_steps = 30->32
Original bitmap score: 0.42972313236326354
Final svg score: 0.5655318367046187
```

| Version | Score | Bitmap Score (Mimic) | SVG Score (Mimic) |
|---|---|---|---|
| 14 | 0.626 | 0.43043661785813436 | 0.6920924696064129 |
| 15 | 0.606 | 0.470562931493742  | 0.6734555091769231 |
| 16 | 0.603 | 0.4897216864586153 | 0.6115771262854257 |
| 17 | 0.625, 0.628 | 0.41999372640056415 | 0.6954702749231211 |
| 18 | None  | 0.3047210546056951 | 0.6567549388044498 |
| 19 | 0.627, 0.632 | 0.36218037075253584 | 0.6712611671713754 |
| 21 | None  | 0.3813717894761453 | 0.649613145463659  |
| 22 | 0.624 | 0.4363026800935096 | 0.6550174730417202 |
| 23 | None | 0.44566469479250975 | 0.6392523522905852 |
| 24 | 0.620 | 0.5532725086267603 | 0.6555382395071472 |
| 25 | None  | 0.5275079692029931 | 0.6604855959343606 |
| 26 | None | 0.44129766133958587 | 0.6610786231381424 |
| 27 | None | 0.33338598137235526 | 0.6765057045624071 |
| 28 | 0.645, 0.643 | 0.40706078202781587 | 0.6912730022532566 |
| 29 | 0.624   | 0.2706269242860204 | 0.6720106723585885 |
| 31 | None | 0.42972313236326354 | 0.5655318367046187 |
