from typing import Callable

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from spikingjelly import visualizing
from spikingjelly.clock_driven.neuron import LIFNode as LIFNode_sj
from spikingjelly.clock_driven.neuron import ParametricLIFNode as PLIFNode_sj
from torch import nn
import matplotlib.pyplot as plt
from modules.surrogate import Rectangle

class softLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        # FOR FIGURE
        self.register_memory('u', 0)

    def forward(self, x: torch.Tensor):

        # Leaky and Integrate
        self.v = self.v * 0.5 + x
        self.u = self.v
        # Spike
        spike = self.surrogate_function(self.v - self.v_threshold)
        # reset
        self.v = self.v - spike * self.v_threshold

        return spike

class hardLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)
        # FOR FIGURE
        self.register_memory('u', 0)

    def forward(self, x: torch.Tensor):

        # Leaky and Integrate
        self.v = self.v * 0.5 + x
        self.u = self.v
        # Spike
        spike = self.surrogate_function(self.v - self.v_threshold)
        # reset
        self.v = self.v * (1 - spike)

        return spike

class XLIF(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, channel=0, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)

        # ADDED Parameters
        self.a = nn.Parameter(torch.tensor(.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # FOR FIGURE
        self.register_memory('s', 0)
        # ADDED Adaptive Reset V
        self.register_memory('m', 0)

    def forward(self, x: torch.Tensor):

        if type(self.m) is not torch.Tensor:
            self.m = torch.zeros_like(x)
        # Leaky and Integrate
        self.v = self.v * 0.5 + x
        # Spike
        spike = self.surrogate_function(self.v - self.v_threshold - self.a.tanh() * x.tanh())
        self.s = spike
        # ADD
        # reset voltage leaky
        self.m = self.m.relu() * torch.sigmoid(self.c * x) + -(-self.m).relu() * (1 - torch.sigmoid(self.c * x))
        # reset voltage integrate
        self.m += spike * torch.sigmoid(x)
        self.m -= (1 - spike) * torch.sigmoid(x)
        # Adaptive reset
        self.v = self.v - spike * (self.v_threshold + torch.sigmoid(self.m) + self.a.tanh() * x.tanh())

        return spike


class ReLU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


class BPTTNeuron(LIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = .0, surrogate_function: Callable = Rectangle(),
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, cupy_fp32_inference)


class PLIFNeuron(PLIFNode_sj):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, cupy_fp32_inference=False, **kwargs):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)

