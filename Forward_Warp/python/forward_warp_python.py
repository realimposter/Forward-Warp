import torch
from torch.nn import Module, Parameter
from torch.autograd import Function

class Forward_Warp_Python:
    @staticmethod
    def forward(im0, flow, interpolation_mode,inpaint_search_radius, inpaint_motion_threshold):
        im1 = torch.zeros_like(im0)
        B = im0.shape[0]
        H = im0.shape[2]
        W = im0.shape[3]
        if interpolation_mode == 0:
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        x = w + flow[b, h, w, 0]
                        y = h + flow[b, h, w, 1]
                        nw = (int(torch.floor(x)), int(torch.floor(y)))
                        ne = (nw[0]+1, nw[1])
                        sw = (nw[0], nw[1]+1)
                        se = (nw[0]+1, nw[1]+1)
                        p = im0[b, :, h, w]
                        if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
                            nw_k = (se[0]-x)*(se[1]-y)
                            ne_k = (x-sw[0])*(sw[1]-y)
                            sw_k = (ne[0]-x)*(y-ne[1])
                            se_k = (x-nw[0])*(y-nw[1])
                            im1[b, :, nw[1], nw[0]] += nw_k*p
                            im1[b, :, ne[1], ne[0]] += ne_k*p
                            im1[b, :, sw[1], sw[0]] += sw_k*p
                            im1[b, :, se[1], se[0]] += se_k*p
        else:
            round_flow = torch.round(flow)
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        x = w + int(round_flow[b, h, w, 0])
                        y = h + int(round_flow[b, h, w, 1])
                        if x >= 0 and x < W and y >= 0 and y < H:
                            im1[b, :, y, x] = im0[b, :, h, w]
        return im1
