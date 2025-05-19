[Version 17, Version 23, Version 26, Version 27, Version 28]

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
bitmap = generate_bitmap(prompt, self.negative_prompt, self.num_inference_steps, self.guidance_scale)
Original bitmap score: 0.2706269242860204
Final svg score: 0.6720106723585885
```

[Version 22, Version 30]

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
| 19 | 0.627 | 0.36218037075253584 | 0.6712611671713754 |
| 21 | None  | 0.3813717894761453 | 0.649613145463659  |
| 22 | 0.624 | 0.4363026800935096 | 0.6550174730417202 |
| 23 | None | 0.44566469479250975 | 0.6392523522905852 |
| 24 | 0.620 | 0.5532725086267603 | 0.6555382395071472 |
| 25 | None  | 0.5275079692029931 | 0.6604855959343606 |
| 26 | None | 0.44129766133958587 | 0.6610786231381424 |
| 27 | None | 0.33338598137235526 | 0.6765057045624071 |
| 28 | TBD  | 0.40706078202781587 | 0.6912730022532566 |
| 29 | TBD   | 0.2706269242860204 | 0.6720106723585885 |
| 31 | None | 0.42972313236326354 | 0.5655318367046187 |
