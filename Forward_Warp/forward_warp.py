import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

import forward_warp_cuda
from .python import Forward_Warp_Python


class forward_warp_function(Function):

    @staticmethod
    def forward(ctx, im0, flow, flowback, interpolation_mode):
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(interpolation_mode in (0, 1))
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        ctx.interpolation_mode = interpolation_mode
        im1 = forward_warp_cuda.forward(im0, flow, flowback, interpolation_mode)
        return im1

class forward_warp(Module):
    def __init__(self, interpolation_mode="Bilinear"):
        super(forward_warp, self).__init__()
        assert(interpolation_mode in ("Bilinear", "Nearest"))
        if(interpolation_mode == "Bilinear"):
            self.interpolation_mode = 0
        else:
            self.interpolation_mode = 1
    def forward(self, im0, flow, flowback):
        return forward_warp_function.apply(im0, flow, flowback, self.interpolation_mode)
