import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import forward_warp_cuda
from .python import Forward_Warp_Python


class forward_warp_function(Function):

    @staticmethod
    def forward(ctx, im0, flow, flowback, infil_iterations, interpolation_mode):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        interpolation_mode: 0 is Bilinear, 1 is Nearest
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(interpolation_mode in (0, 1))
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        assert(flowback.shape[3] == 2)

        ctx.interpolation_mode = interpolation_mode
        ctx.save_for_backward(im0, flow)
        if im0.is_cuda:
            im1 = forward_warp_cuda.forward(im0, flow, flowback, infil_iterations, interpolation_mode)
        else:
            im1 = Forward_Warp_Python.forward(im0, flow, flowback, infil_iterations, interpolation_mode)

        return im1


class forward_warp(Module):

    def __init__(self, interpolation_mode="Bilinear"):
        '''
        Support interpolation mode with Bilinear and Nearest.
        '''
        super(forward_warp, self).__init__()
        assert(interpolation_mode in ("Bilinear", "Nearest"))
        if(interpolation_mode == "Bilinear"):
            self.interpolation_mode = 0
        else:
            self.interpolation_mode = 1

    def forward(self, im0, flow, flowback, infil_iterations):

        return forward_warp_function.apply(im0, flow, flowback, infil_iterations, self.interpolation_mode)
