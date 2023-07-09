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
    scalar_t* flow,
    scalar_t* im1,
    scalar_t* white_im1,
    const int B,
    const int C,
    const int H,
    const int W,
    const GridSamplerInterpolation interpolation_mode) {
    int dilate_radius = 2; // Adjust the size for dilation here. 1 means 3x3 neighborhood.
    CUDA_KERNEL_LOOP(index, total_step) {
        const int b = index / (H * W);
        const int h = (index-b*H*W) / W;
        const int w = index % W;

        // Initialize largest_flow_amplitude and its location
        scalar_t largest_flow_amplitude = -1;
        int largest_loc = -1;

        // Iterate over the neighborhood defined by dilate_radius
        for (int i = max(0, h-dilate_radius); i <= min(H-1, h+dilate_radius); ++i) {
            for (int j = max(0, w-dilate_radius); j <= min(W-1, w+dilate_radius); ++j) {
                const int neighbor_index = b*H*W + i*W + j;
                const scalar_t neighbor_flow_amplitude = hypotf(flow[neighbor_index*2+0], flow[neighbor_index*2+1]);
                if (neighbor_flow_amplitude > largest_flow_amplitude) {
                    largest_flow_amplitude = neighbor_flow_amplitude;
                    largest_loc = neighbor_index;
                }
            }
        }

        // Check if the largest_flow_amplitude is more than 4 greater than current pixel amplitude
        const scalar_t current_flow_amplitude = hypotf(flow[index*2+0], flow[index*2+1]);
        if ((largest_flow_amplitude - current_flow_amplitude) > 3) {
            // Update flow
            flow[index*2+0] = flow[largest_loc*2+0];
            flow[index*2+1] = flow[largest_loc*2+1];
        }

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
                scalar_t* white_im1_p = white_im1+get_im_index(b, 0, y_f, x_f, C, H, W); // added warped white image
                for (int c = 0; c < C; ++c, im0_p+=H*W, im1_p+=H*W, white_im1_p+=H*W){
                    atomicAdd(im1_p,     nw_k*(*im0_p));
                    atomicAdd(im1_p+1,   ne_k*(*im0_p));
                    atomicAdd(im1_p+W,   sw_k*(*im0_p));
                    atomicAdd(im1_p+W+1, se_k*(*im0_p));
                    atomicAdd(white_im1_p,     nw_k); // added warped white image
                    atomicAdd(white_im1_p+1,   ne_k); // added warped white image
                    atomicAdd(white_im1_p+W,   sw_k); // added warped white image
                    atomicAdd(white_im1_p+W+1, se_k); // added warped white image
                }
            }
        } 
        else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            const int x_nearest = static_cast<int>(::round(x));
            const int y_nearest = static_cast<int>(::round(y));
            if (x_nearest >= 0 && x_nearest < W && y_nearest >= 0 && y_nearest < H) {
                const scalar_t* im0_p = im0 + get_im_index(b, 0, h, w, C, H, W);
                scalar_t* im1_p = im1 + get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
                scalar_t* white_im1_p = white_im1 + get_im_index(b, 0, y_nearest, x_nearest, C, H, W); // added warped white image
                for (int c = 0; c < C; ++c, im0_p += H*W, im1_p += H*W, white_im1_p += H*W) {
                    *im1_p = *im0_p;
                    *white_im1_p = 1; // set pixel value to 1 for the warped white image
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
          const scalar_t sw_grad = grad_output[get_im_index(b, c, y_c, x_f, C, H, W)];
          const scalar_t se_grad = grad_output[get_im_index(b, c, y_c, x_c, C, H, W)];
          const scalar_t p = im0[get_im_index(b, c, h, w, C, H, W)];
          atomicAdd(im0_grad_p, nw_k*nw_grad);
          atomicAdd(im0_grad_p, ne_k*ne_grad);
          atomicAdd(im0_grad_p, sw_k*sw_grad);
          atomicAdd(im0_grad_p, se_k*se_grad);
          flow_grad_x -= (y_c-y)*p*nw_grad;
          flow_grad_y -= (x_c-x)*p*nw_grad;
          flow_grad_x += (y_c-y)*p*ne_grad;
          flow_grad_y -= (x-x_f)*p*ne_grad;
          flow_grad_x -= (y-y_f)*p*sw_grad;
          flow_grad_y += (x_c-x)*p*sw_grad;
          flow_grad_x += (y-y_f)*p*se_grad;
          flow_grad_y += (x-x_f)*p*se_grad;
        }
        flow_grad[index*2+0] = flow_grad_x;
        flow_grad[index*2+1] = flow_grad_y;
      }
    } 
    else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      const int x_nearest = static_cast<int>(::round(x));
      const int y_nearest = static_cast<int>(::round(y));
      if(x_nearest>=0 && x_nearest<W && y_nearest>=0 && y_nearest<H){
        scalar_t* im0_grad_p = im0_grad+get_im_index(b, 0, h, w, C, H, W);
        const scalar_t* im1_grad_p = grad_output+get_im_index(b, 0, y_nearest, x_nearest, C, H, W);
        for (int c = 0; c < C; ++c, im0_grad_p += H*W, im1_grad_p += H*W) {
            *im0_grad_p = *im1_grad_p;
        }
      }
    }
  }
}

at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const GridSamplerInterpolation interpolation_mode) {
  auto im1 = at::zeros_like(im0);
  auto white_im1 = at::ones_like(im0); // create an all-white image of same size as im0
  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;
  AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] {
    forward_warp_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      im0.data<scalar_t>(),
      flow.data<scalar_t>(),
      im1.data<scalar_t>(),
      white_im1.data<scalar_t>(), // added warped white image
      B, C, H, W,
      interpolation_mode);
  }));
  // Divide warped main image by warped white image
  im1.div_(white_im1-1);
  return im1;
}

std::vector<at::Tensor> forward_warp_cuda_backward(
    const at::Tensor grad_output,
    const at::Tensor im0, 
    const at::Tensor flow,
    const GridSamplerInterpolation interpolation_mode) {
  auto im0_grad = at::zeros_like(grad_output);
  auto flow_grad = at::empty_like(flow);
  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "forward_warp_backward_cuda", ([&] {
    forward_warp_cuda_backward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      grad_output.data<scalar_t>(),
      im0.data_ptr<scalar_t>(),
      flow.data<scalar_t>(),
      im0_grad.data<scalar_t>(),
      flow_grad.data<scalar_t>(),
      B, C, H, W,
      interpolation_mode);
  }));

  return {im0_grad, flow_grad};
}
