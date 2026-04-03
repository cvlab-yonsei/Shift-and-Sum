import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from quant.quantizer import UniformAffineQuantizer, GroupAffineQuantizer, PerTokenQuantizer, PerGroupQuantizer, PerTokenLog2Quantizer, Shift_and_Sum_Quantizer

from models.quant import Phi
import numpy as np
import copy

# For conv, linear operations
class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.disable_act_quant = False

        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        # initialize quantizer
        if weight_quant_params['n_bits'] >= 6 or self.weight.size(1) % 128 != 0:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        else:
            self.weight_quantizer = GroupAffineQuantizer(**weight_quant_params)

        if not self.disable_act_quant:
            if act_quant_params['n_bits'] >= 6 or self.weight.size(1) % 128 != 0:
                self.act_quantizer = PerTokenQuantizer(**act_quant_params)
            else:
                self.act_quantizer = PerGroupQuantizer(**act_quant_params)

        # Exception for phis
        self.resi_ratio = org_module.resi_ratio if isinstance(org_module, Phi) else None
        self.patch_nums = []
        self.init_patch_res = False

    def compute_patch_res(self, input: torch.Tensor):
        if self.fwd_func == F.conv2d:
            self.patch_nums.append(input.size(2) * input.size(3))
        elif self.fwd_func == F.linear:
            if input.ndim == 3:
                self.patch_nums.append(input.size(1))
            elif input.ndim == 2:
                self.patch_nums.append(1)
        self.patch_nums = sorted(list(set(self.patch_nums)))

    def forward(self, input: torch.Tensor):
        if not self.init_patch_res:
            self.compute_patch_res(input)
            
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        if self.use_act_quant and not self.disable_act_quant:
            input = self.act_quantizer(input)
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        if self.resi_ratio is not None:
            out = out.mul(self.resi_ratio) + input.mul(1 - self.resi_ratio)
            
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


class QuantMatMul(nn.Module):
    def __init__(self, mode: str = 'matmul1', matmul_params: dict = {}):
        super().__init__()
        self.mode = mode
        self.disable_act_quant = False

        self.use_A_quant = False
        self.use_B_quant = False

        def build_params(**overrides):
            params = copy.deepcopy(matmul_params)
            params.update(overrides)
            return params

        if self.mode == 'matmul1':
            if matmul_params['n_bits'] >= 6:
                self.A_quantizer = PerTokenQuantizer(**build_params())
                self.B_quantizer = PerTokenQuantizer(**build_params())
            else:
                self.A_quantizer = PerGroupQuantizer(**build_params(dim=(0, 1, 2)))
                self.B_quantizer = PerGroupQuantizer(**build_params(dim=(0, 1, 3)))

        elif self.mode == 'matmul2':
            self.A_quantizer = PerTokenLog2Quantizer(**build_params())
            self.B_quantizer = Shift_and_Sum_Quantizer(**build_params(dim=(0, 1)))

        self.theta = 1.0
        
        self.A_patch_nums = []
        self.B_patch_nums = []
        self.init_patch_res = False

    def compute_patch_res(self, A: torch.Tensor, B: torch.Tensor):
        if self.mode == 'matmul1':
            self.A_patch_nums.append(A.size(2)) # [s1]
            self.B_patch_nums.append(B.size(3)) # [s2]
        elif self.mode == 'matmul2':
            self.A_patch_nums.append(A.size(2)) # [s1]
            self.B_patch_nums.append(B.size(2)) # [s2]
        self.A_patch_nums = sorted(list(set(self.A_patch_nums)))
        self.B_patch_nums = sorted(list(set(self.B_patch_nums)))

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        if not self.init_patch_res:
            self.compute_patch_res(A, B)

        if self.mode == 'matmul1':
            if self.use_A_quant:
                A = self.A_quantizer(A)
            if self.use_B_quant:
                B = self.B_quantizer(B)

        elif self.mode == 'matmul2':
            if isinstance(self.B_quantizer, Shift_and_Sum_Quantizer):
                n = torch.log2(A.mean(dim=2) / self.theta).ceil().clamp(min=0, max=3)
                if self.use_A_quant:
                    A = self.A_quantizer(A)
                if self.use_B_quant:
                    B = self.B_quantizer(B, n)
            else:
                if self.use_A_quant:
                    A = self.A_quantizer(A)
                if self.use_B_quant:
                    B = self.B_quantizer(B)
        out = A @ B
        return out

    def set_quant_state(self, A_quant: bool = False, B_quant: bool = False):
        self.use_A_quant = A_quant
        self.use_B_quant = B_quant