#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "forward_warp.h"
using at::native::detail::GridSamplerInterpolation;

static __forceinline__ __device__ 
int get_channel_index(
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
__global__ void forward_mask_kernel(
    const int total_step,
    const scalar_t* im0,
    scalar_t* flow,
    scalar_t* im1,
    const int B,
    const int C,
    const int H,
    const int W) {
    const int dilate_radius = 1;
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
    const int x_f = static_cast<int>(::floor(x));
    const int y_f = static_cast<int>(::floor(y));
    const int x_c = x_f + 1;
    const int y_c = y_f + 1;
    if(x_f>=0 && x_c<W && y_f>=0 && y_c<H){
      const scalar_t nw_k = (x_c - x) * (y_c - y);
      const scalar_t ne_k = (x - x_f) * (y_c - y);
      const scalar_t sw_k = (x_c - x) * (y - y_f);
      const scalar_t se_k = (x - x_f) * (y - y_f);
      const scalar_t* im0_p = im0+get_channel_index(b, 0, h, w, C, H, W);
      scalar_t* im1_p = im1+get_channel_index(b, 0, y_f, x_f, C, H, W);
      for (int c = 0; c < C; ++c, im0_p+=H*W, im1_p+=H*W){
          atomicAdd(im1_p,     nw_k*(*im0_p));
          atomicAdd(im1_p+1,   ne_k*(*im0_p));
          atomicAdd(im1_p+W,   sw_k*(*im0_p));
          atomicAdd(im1_p+W+1, se_k*(*im0_p));
      }
    }
  }
}

template <typename scalar_t>
__global__ void dialate_kernel(
    const int total_step,
    scalar_t* im1,
    const int radius,
    const int B,
    const int C,
    const int H,
    const int W) {
  CUDA_KERNEL_LOOP(index, total_step) {
    const int b = index / (H * W * C);
    const int c = (index / (H * W)) % C;
    const int h = (index / W) % H;
    const int w = index % W;

    bool set_nan = false;
    for(int i = max(0, h - radius); i <= min(H - 1, h + radius); i++) {
      for(int j = max(0, w - radius); j <= min(W - 1, w + radius); j++) {
        const scalar_t* pixel = im1 + get_channel_index(b, c, i, j, C, H, W);
        if(*pixel == 0) {
          set_nan = true;
          break;
        }
      }
      if(set_nan) break;
    }

    if(set_nan) {
      scalar_t* target_pixel = im1 + get_channel_index(b, 0, h, w, C, H, W);
      for (int c = 0; c < C; ++c, target_pixel += H*W) {
          *target_pixel = NAN;
      }

    }
  }
}

template <typename scalar_t>
__global__ void inpaint_nan_pixels_kernel(
    const int total_step,
    scalar_t* im0,
    const scalar_t* flow,
    const int radius,
    const scalar_t motion_threshold,
    const int B,
    const int C,
    const int H,
    const int W) {

    __shared__ int nan_count;

    for (int iteration = 0; iteration < 64; ++iteration) {
        nan_count = 0;
        CUDA_KERNEL_LOOP(index, total_step) {
            const int b = index / (H * W);
            const int h = (index-b*H*W) / W;
            const int w = index % W;

            // get flow vector
            const scalar_t x1 = (scalar_t)w + flow[index*2+0];
            const scalar_t y1 = (scalar_t)h + flow[index*2+1];

            // get current pixel
            const scalar_t* im0_p = im0 + get_channel_index(b, 0, h, w, C, H, W);

            // make sure pixel is not NaN, if it is end the loop
            if (!isnan(*im0_p)) continue;

            // since pixel is NaN, increment nan_count
            nan_count++;

            // foreach pixel in a radius of "radius" around the current pixel index
            int neighbor_count = 0;
            scalar_t sum[3] = {0}; // Initialize array with zeros
            for (int neighbor_w = max(0, w - radius); neighbor_w <= min(W - 1, w + radius); ++neighbor_w) {
                for (int neighbor_h = max(0, h - radius); neighbor_h <= min(H - 1, h + radius); ++neighbor_h) {
                    // get neighbor index
                    const int neighbor_index = get_channel_index(b, 0, neighbor_h, neighbor_w, C, H, W);

                    // if neighbor pixel is Nan, move on to the next neighbor
                    if (isnan(*(im0 + neighbor_index))) continue;

                    // get neighbor flow
                    const int neighbor_flow_index = get_channel_index(b, 0, neighbor_h, neighbor_w, 2, H, W);
                    const scalar_t x2 = (scalar_t)w + flow[neighbor_flow_index*2+0];
                    const scalar_t y2 = (scalar_t)h + flow[neighbor_flow_index*2+1];

                    // compare neighbor flow to current pixel flow
                    const scalar_t flowDiff = abs(x1 - x2) + abs(y1 - y2);

                    // if the flows are different, move on to the next neighbor.
                    if (flowDiff > motion_threshold) continue;

                    // add the neighbor pixel value to the sum
                    scalar_t* im1_p = im0 + neighbor_index;
                    for (int c = 0; c < C; ++c, im1_p += H*W) {
                        sum[c] += *im1_p;
                    }

                    // increment the count of valid neighbors
                    neighbor_count++;
                }
            }

            // if any valid neighbors were found, average their values and use that to replace the current pixel
            if (neighbor_count > 0) {
                scalar_t* im0_p = im0 + get_channel_index(b, 0, h, w, C, H, W);
                for (int c = 0; c < C; ++c, im0_p += H*W) {
                    *im0_p = sum[c] / neighbor_count;
                }
            }
        }

        __syncthreads();  

        // Break out of the loop if no NaN pixels are found
        if (nan_count == 0) break;
    }
}



at::Tensor forward_warp_cuda_forward(
    const at::Tensor input_image,
    const at::Tensor flow,
    const at::Tensor flowback,
    const GridSamplerInterpolation interpolation_mode,
    const int inpaint_search_radius,
    const float inpaint_motion_threshold) {
  auto output_image = at::zeros_like(input_image);
  auto white = at::ones_like(input_image);
  auto mask = at::zeros_like(input_image);
  const int B = input_image.size(0);
  const int C = input_image.size(1);
  const int H = input_image.size(2);
  const int W = input_image.size(3);
  const int total_step = B * H * W;
  AT_DISPATCH_FLOATING_TYPES(input_image.scalar_type(), "forward_warp_forward_cuda", ([&] {
    
    /////// WARP BACKWARDS //////////
    back_warp_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      input_image.data_ptr<scalar_t>(),
      flowback.data_ptr<scalar_t>(),
      output_image.data_ptr<scalar_t>(),
      B, C, H, W);

    /////// CREATE MASK FROM FORWARD WARP HOLES //////////
    forward_mask_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      white.data_ptr<scalar_t>(),
      flow.data_ptr<scalar_t>(),
      mask.data_ptr<scalar_t>(),
      B, C, H, W);

    /////// DIALATE MASK //////////
    dialate_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      mask.data_ptr<scalar_t>(),
      1,
      B, C, H, W);

    //////// MASK BACKWARP WITH FORWARD WARP HOLES////////
    output_image = at::where(isnan(mask), mask, output_image);

    ///// INPAINT HOLES //////////
    inpaint_nan_pixels_kernel<scalar_t>
    <<<GET_BLOCKS(total_step), CUDA_NUM_THREADS>>>(
      total_step,
      output_image.data_ptr<scalar_t>(),
      flowback.data_ptr<scalar_t>(),
      inpaint_search_radius,
      inpaint_motion_threshold,
      B, C, H, W);

  }));
  return output_image;
}
