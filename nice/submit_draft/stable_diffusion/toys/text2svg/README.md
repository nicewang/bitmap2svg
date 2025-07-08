### 1. Design Ideas

### 2. Versions
- [1] 
- [2,2_opt,2_opt_2,[2_opt_3,2_opt_3_perceploss_vgg,2_opt_3_lpips],2_tpu]:
    - cuda out of memory: 2,2_opt,2_opt_2
    - runable: 2_opt_3
    - 2_opt_3_perceploss_vgg: not useful
    - 2_opt_3_lpips:
        - Increasing lambda (e.g. 0.2, 0.5): will make the model focus more on accurate color and brightness matching, possibly sacrificing some structural flexibility.
        - Decreasing lambda (e.g. 0.05, 0.01): will allow the model to more freely adjust the colors to better match the texture and shape, but may cause the final SVG to have a different tone than the original image.

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