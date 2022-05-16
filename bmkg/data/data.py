import torch

from .._data import *


class TripleDataBatchGPU:
    h: torch.Tensor
    r: torch.Tensor
    t: torch.Tensor
    cpu: TripleDataBatch

    def __init__(self, *args):
        if len(args) == 1:
            data = args[0]
            self.cpu = data

            h = torch.from_numpy(data.h)
            r = torch.from_numpy(data.r)
            t = torch.from_numpy(data.t)
            h = h.cuda()
            r = r.cuda()
            t = t.cuda()

            self.h = h
            self.r = r
            self.t = t
        else:
            h, r, t = args
            self.h = h
            self.r = r
            self.t = t
