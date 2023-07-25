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
    int dilate_radius = 24; // Adjust the size for dilation here. 1 means 3x3 neighborhood.
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
__global__ void back_warp_kernel(
    const int total_step,
    const scalar_t* im0,
    scalar_t* flow,
    scalar_t* im1,
    const int B,
    const int C,
    const int H,
    const int W) {
    CUDA_KERNEL_LOOP(index, total_step) {
        const int b = index / (H * W);
        const int h = (index-b*H*W) / W;
        const int w = index % W;
        
        const scalar_t x = (scalar_t)w + flow[index*2+0];
        const scalar_t y = (scalar_t)h + flow[index*2+1];

        //get im1 value from im0 using flow, leaving holes as NaN
        if (x < 0 || y < 0 || x >= W || y >= H) {
            // Out of bound, leave as NaN
            for (int c = 0; c < C; ++c) {
                im1[b * (H * W * C) + c * (H * W) + h * W + w] = NAN;
            }
        } else {
            // Bilinear interpolation
            const int x1 = static_cast<int>(::floor(x));
            const int y1 = static_cast<int>(::floor(y));
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            const scalar_t dist_x = x - x1;
            const scalar_t dist_y = y - y1;

            for (int c = 0; c < C; ++c) {
                // Compute weights
                scalar_t w1 = (1 - dist_x) * (1 - dist_y);
                scalar_t w2 = dist_x * (1 - dist_y);
                scalar_t w3 = (1 - dist_x) * dist_y;
                scalar_t w4 = dist_x * dist_y;

                // Fetch pixel values
                scalar_t p1 = (x1 < W && y1 < H) ? im0[b * (H * W * C) + c * (H * W) + y1 * W + x1] : NAN;
                scalar_t p2 = (x2 < W && y1 < H) ? im0[b * (H * W * C) + c * (H * W) + y1 * W + x2] : NAN;
                scalar_t p3 = (x1 < W && y2 < H) ? im0[b * (H * W * C) + c * (H * W) + y2 * W + x1] : NAN;
                scalar_t p4 = (x2 < W && y2 < H) ? im0[b * (H * W * C) + c * (H * W) + y2 * W + x2] : NAN;

                // Check for NaN values in the fetched pixels
                if (isnan(p1) || isnan(p2) || isnan(p3) || isnan(p4)) {
                    im1[b * (H * W * C) + c * (H * W) + h * W + w] = NAN;
                } else {
                    // Compute interpolated pixel value
                    scalar_t px_val = p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4;
                    im1[b * (H * W * C) + c * (H * W) + h * W + w] = px_val;
                }
            }
        }
    }
}


template <typename scalar_t>
__global__ void inpaint_nan_pixels_kernel(
    scalar_t* im1,
    const scalar_t* flowback,
    const int B,
    const int C,
    const int H,
    const int W) {
    const int total_step = B * C * H * W;
    const int radius = 4;  // Add this line to define radius

    for (int iteration = 0; iteration < 24; ++iteration) {
        bool has_nan = false;  // Track if NaN pixels are found in the iteration

        CUDA_KERNEL_LOOP(index, total_step) {
            if (!isnan(im1[index])) continue;

            const int b = index / (C * H * W);
            const int c = (index - b * C * H * W) / (H * W);
            const int h = (index - b * C * H * W - c * H * W) / W;
            const int w = index % W;

            scalar_t sum = 0;
            int count = 0;

            // Update loop bounds to consider a radius
            for (int i = max(0, h - radius); i <= min(H - 1, h + radius); ++i) {
                for (int j = max(0, w - radius); j <= min(W - 1, w + radius); ++j) {
                    const int neighbor_index = get_im_index(b, c, i, j, C, H, W);
                    if (!isnan(im1[neighbor_index])) {

                        const scalar_t flow_x = flowback[index*2+0];
                        const scalar_t flow_y = flowback[index*2+1];

                        const scalar_t flow_x2 = flowback[neighbor_index*2+0];
                        const scalar_t flow_y2 = flowback[neighbor_index*2+1];

                        scalar_t flowback_diff = abs(flow_x - flow_x2) + abs(flow_y - flow_y2);

                        if (flowback_diff < 0.5){
                          sum += im1[neighbor_index];
                          count=count+1;
                        }
                    }
                }
            }

            if (count > 0) im1[index] = sum / count;
            else has_nan = true;  // Set has_nan to true if NaN pixels are found
        }

        __syncthreads();  // Add this line to synchronize threads before starting the next iteration

        // Break out of the loop if no NaN pixels are found
        if (!has_nan) break;
    }
}

at::Tensor forward_warp_cuda_forward(
    const at::Tensor im0,
    const at::Tensor flow,
    const at::Tensor flowback,
    const GridSamplerInterpolation interpolation_mode) {
  auto im1 = at::zeros_like(im0);
  auto im2 = at::zeros_like(im0);
  auto white_im1 = at::ones_like(im0); // create an all-white image of same size as im0
  const int B = im0.size(0);
  const int C = im0.size(1);
  const int H = im0.size(2);
  const int W = im0.size(3);
  const int total_step = B * H * W;
  AT_DISPATCH_FLOATING_TYPES(im0.scalar_type(), "forward_warp_forward_cuda", ([&] {

    /////////// FORWARD WARP ////////////
    forward_warp_cuda_forward_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      im0.data<scalar_t>(),
      flow.data<scalar_t>(),
      im1.data<scalar_t>(),
      white_im1.data<scalar_t>(), // added warped white image
      B, C, H, W,
      interpolation_mode);

    // Divide warped main image by warped white image
    im1.div_(white_im1-1);

    /////// WARPP BACKWARDS //////////
    back_warp_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      im0.data_ptr<scalar_t>(),
      flowback.data_ptr<scalar_t>(),
      im2.data_ptr<scalar_t>(),
      B, C, H, W);

    //////// mask im2 by adding im1 NaNs ////////
    auto nan_mask = at::isnan(im1);
    im2 = at::where(nan_mask, im1, im2);

    /////// INPAINTING //////////
    inpaint_nan_pixels_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      im2.data_ptr<scalar_t>(),
      flowback.data_ptr<scalar_t>(),
      B, C, H, W);

  }));
  return im2;
}
