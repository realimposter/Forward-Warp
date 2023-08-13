#include <torch/torch.h>
#include <vector>

#include "forward_warp.h"
using at::native::detail::GridSamplerInterpolation;

at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const at::Tensor flowback,
    const GridSamplerInterpolation interpolation_mode,
    const int inpaint_search_radius,
    const float inpaint_motion_threshold,
    const int max_iterations,
    const int mask_dilation,
    const int dilate_radius);

// Because of the incompatible of Pytorch 1.0 && Pytorch 0.4, we have to annotation this.
#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor forward_warp_forward(
    const at::Tensor im0, 
    const at::Tensor flow,
    const at::Tensor flowback,
    const int interpolation_mode,
    const int inpaint_search_radius,
    const float inpaint_motion_threshold,
    const int max_iterations,
    const int mask_dilation,
    const int dilate_radius){
  // CHECK_INPUT(im0);
  // CHECK_INPUT(flow);
  return forward_warp_cuda_forward(im0, flow, flowback, (GridSamplerInterpolation)interpolation_mode, inpaint_search_radius, inpaint_motion_threshold,max_iterations,mask_dilation,dilate_radius);
}

PYBIND11_MODULE(
    TORCH_EXTENSION_NAME, 
    m){
  m.def("forward", &forward_warp_forward, "forward warp forward (CUDA)");
}
