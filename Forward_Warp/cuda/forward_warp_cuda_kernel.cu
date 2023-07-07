#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "forward_warp.h"
using at::native::detail::GridSamplerInterpolation;

static __forceinline__ __device__ 
int get_im_index(
    const int b,
    const int c,
    const int h,
    const int w,
    const size_t C,
    const size_t H,
    const size_t W) {
  return b*C*H*W + c*H*W + h*W + w;
}

template <typename scalar_t>
__global__ void forward_warp_cuda_forward_kernel(
    const int total_step,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im1,
    scalar_t* sort,
    const int B,
    const int C,
    const int H,
    const int W,
    const GridSamplerInterpolation interpolation_mode) {
  CUDA_KERNEL_LOOP(index, total_step) {
    const int b = index / (H * W);
    const int h = (index-b*H*W) / W;
    const int w = index % W;
    const scalar_t x = (scalar_t)w + flow[index*2+0];
    const scalar_t y = (scalar_t)h + flow[index*2+1];
    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      const int x_f = static_cast<int>(::floor(x));
      const int y_f = static_cast<int>(::floor(y));
      const int x_c = x_f + 1;
      const int y_c = y_f + 1;
      if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
        const scalar_t nw_k = (x_c - x) * (y_c - y);
        const scalar_t ne_k = (x - x_f) * (y_c - y);
        const scalar_t sw_k = (x_c - x) * (y - y_f);
        const scalar_t se_k = (x - x_f) * (y - y_f);
        const scalar_t* im0_p = im0+get_im_index(b, 0, h, w, C, H, W);
        scalar_t* im1_p = im1+get_im_index(b, 0, y_f, x_f, C, H, W);
        scalar_t* sort_p = sort+get_im_index(b, 0, y_f, x_f, C, H, W);
        for (int c = 0; c < C; ++c, im0_p+=H*W, im1_p+=H*W, sort_p+=H*W){
          const scalar_t curr_sort = *sort_p;
          if (curr_sort < nw_k) {
            *sort_p = nw_k;
            *im1_p = nw_k * (*im0_p);
          }
          if (curr_sort < ne_k) {
            *sort_p = ne_k;
            *(im1_p+1) = ne_k * (*im0_p);
          }
          if (curr_sort < sw_k) {
            *sort_p = sw_k;
            *(im1_p+W) = sw_k * (*im0_p);
          }
          if (curr_sort < se_k) {
            *sort_p = se_k;
            *(im1_p+W+1) = se_k * (*im0_p);
          }
        }
      }
    } 
    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      const int x_nearest = static_cast<int>(::round(x));
      const int y_nearest = static_cast<int>(::round(y));
      if(x_nearest>=0 && x_nearest<W && y_nearest>=0 && y_nearest<H){
        const scalar_t* im0_p = im0+get_im_index(b, 0, h, w, C, H, W);
        scalar_t* im1_p = im1+get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
        scalar_t* sort_p = sort+get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
        for (int c = 0; c < C; ++c, im0_p += H*W, im1_p += H*W, sort_p += H*W) {
          const scalar_t curr_sort = *sort_p;
          if (curr_sort < 1) {
            *sort_p = 1;
            *im1_p = *im0_p;
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void forward_warp_cuda_backward_kernel(
    const int total_step,
    const scalar_t* grad_output,
    const scalar_t* im0,
    const scalar_t* flow,
    scalar_t* im0_grad,
    scalar_t* flow_grad,
    const int B,
    const int C,
    const int H,
    const int W,
    const GridSamplerInterpolation interpolation_mode) {
  CUDA_KERNEL_LOOP(index, total_step) {
    const int b = index / (H * W);
    const int h = (index-b*H*W) / W;
    const int w = index % W;
    const scalar_t x = (scalar_t)w + flow[index*2+0];
    const scalar_t y = (scalar_t)h + flow[index*2+1];
    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      const int x_f = static_cast<int>(::floor(x));
      const int y_f = static_cast<int>(::floor(y));
      const int x_c = x_f + 1;
      const int y_c = y_f + 1;
      if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
        const scalar_t nw_k = (x_c - x) * (y_c - y);
        const scalar_t sw_k = (x_c - x) * (y - y_f);
        const scalar_t ne_k = (x - x_f) * (y_c - y);
        const scalar_t se_k = (x - x_f) * (y - y_f);
        scalar_t flow_grad_x = 0;
        scalar_t flow_grad_y = 0;
        scalar_t* im0_grad_p = im0_grad+get_im_index(b, 0, h, w, C, H, W);
        for (int c = 0; c < C; ++c, im0_grad_p+=H*W){
          const scalar_t nw_grad = grad_output[get_im_index(b, c, y_f, x_f, C, H, W)];
          const scalar_t ne_grad = grad_output[get_im_index(b, c, y_f, x_c, C, H, W)];
          const scalar_t sw_grad = grad_output[get_im_index(b, c, y_c,
