import math

import numpy as np

from spikingjelly.clock_driven import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x

class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        # p ~ T
        self.p = nn.Parameter(torch.ones(8, 1, 1, 1, 1, device=device))
    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y

class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y



class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

# add
class aLIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(aLIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.gamma = gamma

    def forward(self, x):
        mem_v = []

        mem = 0

        N, T, C, H, W = x.shape

        m = torch.zeros_like(x)

        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            m[:, t, ...] = m[:, t, ...].relu() * torch.sigmoid(self.c * x[:, t, ...]) + -(-m[:, t, ...]).relu() * (1 - torch.sigmoid(self.c * x[:, t, ...]))
            spike = self.heaviside(mem - 1 - self.a.tanh() * x[:, t, ...].tanh(), self.gamma)
            m[:, t, ...] += spike * torch.sigmoid(x[:, t, ...])
            m[:, t, ...] -= (1 - spike) * torch.sigmoid(x[:, t, ...])
            mem = mem - spike * (1 + torch.sigmoid(m[:, t, ...]) + self.a.tanh() * x[:, t, ...].tanh())
            mem_v.append(spike)

        return torch.stack(mem_v, dim=1)
# add

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.pre_spike_mem = []

    def forward(self, x):
        mem_v = []
        # _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            # _mem.append(mem.detach().cpu().clone())
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * 1 - spike
            mem_v.append(spike)
        # self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

class VGGSNN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            Layer(64, 128, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            pool,
            Layer(128, 256, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            Layer(256, 256, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            pool,
            Layer(256, 512, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            pool,
            Layer(512, 512, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            Layer(512, 512, 3, 1, 1),
            aLIFSpike(tau=self.tau),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x




def MH(x):
    return (0.75 - x.pow(2)) / (0.75 * 0.75 * math.sqrt(4.71225)) * torch.exp(-x.pow(2) / 1.5)

def Swish(x):
    return x / (1 + torch.exp(-0.5 * x))

class rectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return surrogate.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None

