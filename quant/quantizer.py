import torch
import torch.nn as nn
from scipy.stats import norm


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, dim: int = None, scale_method: tuple = ('max', 0), leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = dim
        self.scale_method, self.eps = scale_method
        self.delta = None
        self.zero_point = None
        self.leaf_param = leaf_param
        self.inited = True

        self.upper_bound = None
        self.lower_bound = None

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.delta is None or self.zero_point is None:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.dim)
            elif self.leaf_param:
                self.act_momentum_update(x)

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        
        return x_dequant

    def act_momentum_update(self, x, act_range_momentum=0.95):
        if self.dim is None:
            if x.numel() > 1000000:
                sample_ratio = 1000000/x.numel()
                x_sample = torch.randn_like(x) > float(norm.ppf(1 - sample_ratio))
                x_sample = x[x_sample]
            else:
                x_sample = x
            lower_bound, upper_bound = torch.quantile(x_sample.float(), self.eps), torch.quantile(x_sample.float(), 1 - self.eps)
        else:
            dims = [1 for _ in range(x.ndim)]; dims[self.dim] = -1
            x_sample = x.movedim(self.dim, 0).flatten(1)
            lower_bound, upper_bound = torch.quantile(x_sample.float(), self.eps, dim=1), torch.quantile(x_sample.float(), 1 - self.eps, dim=1)
            lower_bound, upper_bound = lower_bound.view(*dims), upper_bound.view(*dims)
            
        self.lower_bound = self.lower_bound * act_range_momentum + lower_bound * (1 - act_range_momentum)
        self.upper_bound = self.upper_bound * act_range_momentum + upper_bound * (1 - act_range_momentum)

        self.delta = torch.clamp((self.upper_bound - self.lower_bound) / (self.n_levels - 1), min=1e-6)
        self.zero_point = torch.clamp((-self.lower_bound / self.delta).round(), 0, self.n_levels - 1)

    def init_quantization_scale(self, x: torch.Tensor, dim: int = None):
        if dim is None:
            if self.scale_method == 'max':
                x_min = x.min().float()
                x_max = x.max().float()
            elif self.scale_method == 'percentile':
                if x.numel() > 1000000: # quantile() cannot get too many elements
                    sample_ratio = 1000000/x.numel()
                    x_sample = torch.randn_like(x) > float(norm.ppf(1 - sample_ratio))
                    x_sample = x[x_sample]
                else:
                    x_sample = x
                x_min = torch.quantile(x_sample.float(), self.eps)
                x_max = torch.quantile(x_sample.float(), 1 - self.eps)
            else:
                raise NotImplementedError
        else:
            if self.scale_method == 'max':
                x_sample = x.movedim(self.dim, 0).flatten(1)
                x_min = x_sample.amin(dim=1).float()
                x_max = x_sample.amax(dim=1).float()
            elif self.scale_method == 'percentile':
                x_sample = x.movedim(self.dim, 0).flatten(1)
                x_min = torch.quantile(x_sample.float(), self.eps, dim=1)
                x_max = torch.quantile(x_sample.float(), 1-self.eps, dim=1)

        delta = (x_max - x_min) / (self.n_levels - 1)
        delta[delta < 1e-6] = 1e-6
        zero_point = (-x_min / delta).round()
        zero_point = torch.clamp(zero_point, 0, self.n_levels - 1)
        self.upper_bound, self.lower_bound = x_max, x_min

        if dim is not None:
            dims = [1 for _ in range(x.ndim)]; dims[dim] = -1
            delta, zero_point = delta.view(*dims), zero_point.view(*dims)
            self.upper_bound, self.lower_bound = self.upper_bound.view(*dims), self.lower_bound.view(*dims)

        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class GroupAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 4, dim: tuple = (0, 1), scale_method: tuple = ('max', 0.0), leaf_param: bool = False):
        super(GroupAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 8
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.group_size = 128
        self.dim = (0, 1)
        self.scale_method, self.eps = scale_method
        self.leaf_param = False
        self.inited = False

        self.delta = None
        self.zero_point = None

    def forward(self, x: torch.Tensor):
        G = x.size(1) // self.group_size
        x_group = x.view(x.size(0), G, self.group_size, *x.size()[2:])

        if not self.inited:
            self.delta, self.zero_point = self.init_quantization_scale(x_group)
            self.inited = True
        
        x_int = round_ste(x_group / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        x_out = x_dequant.view_as(x)

        return x_out

    def init_quantization_scale(self, x_group: torch.Tensor):
        reduce_dims = [i for i in range(x_group.ndim) if i not in self.dim]
        x_min = x_group.amin(dim=reduce_dims, keepdim=True).float()
        x_max = x_group.amax(dim=reduce_dims, keepdim=True).float()

        delta = (x_max - x_min) / (self.n_levels - 1)
        delta[delta < 1e-6] = 1e-6
        zero_point = (-x_min / delta).round()
        zero_point = torch.clamp(zero_point, 0, self.n_levels - 1)

        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class PerTokenQuantizer(nn.Module):
    def __init__(self, n_bits: int = 6, dim: tuple = None, scale_method: tuple = ('max', 0), leaf_param: bool = True):
        super(PerTokenQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = dim
        self.fc2 = False

    def forward(self, x: torch.Tensor):
        reduce_dims = [i for i in range(x.ndim) if i not in self.dim]
        if len(x.size()) == 2:
            reduce_dims = [1]

        M = x.amax(dim=reduce_dims, keepdim=True)
        m = x.amin(dim=reduce_dims, keepdim=True)

        if self.fc2:
            delta_l = torch.clamp(-m / (self.n_levels / 2 - 1), min=1e-6)
            delta_u = torch.clamp(M / (self.n_levels / 2 - 1), min=1e-6)

            x_neg = torch.where(x <= 0, x, torch.zeros_like(x))
            x_pos = torch.where(x > 0, x, torch.zeros_like(x))

            x_neg_int = round_ste(x_neg / delta_l)
            x_pos_int = round_ste(x_pos / delta_u)

            x_neg_quant = torch.clamp(x_neg_int, 1 - self.n_levels / 2, 0)
            x_pos_quant = torch.clamp(x_pos_int, 0, self.n_levels / 2 - 1)

            x_dequant = x_neg_quant * delta_l + x_pos_quant * delta_u
        else:
            delta = torch.clamp((M - m) / (self.n_levels - 1), min=1e-6)
            zero_point = -torch.round(m / delta).detach()
            
            x_int = round_ste(x / delta) + zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

        return x_dequant


class PerGroupQuantizer(nn.Module):
    def __init__(self, n_bits: int = 4, dim: tuple = None, scale_method: tuple = ('max', 0), leaf_param: bool = True):
        super(PerGroupQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = dim
        self.group_size = 64

        self.fc2 = False

    def forward(self, x: torch.Tensor):
        reduce_dims = [i for i in range(x.ndim) if i not in self.dim]
        if len(x.size()) == 2:
            reduce_dims = [1]

        x_group = x
        for dim in reduce_dims:
            C = x.size(dim)
            G = C // self.group_size
            new_shape = [*x.size()[:dim], G, self.group_size, *x.size()[dim+1:]]
            x_group = x_group.view(*new_shape)
            reduce_dims[reduce_dims.index(dim)] += 1
        
        M = x_group.amax(dim=reduce_dims, keepdim=True)
        m = x_group.amin(dim=reduce_dims, keepdim=True)

        if self.fc2:
            delta_l = torch.clamp(-m / (self.n_levels / 2 - 1), min=1e-6)
            delta_u = torch.clamp(M / (self.n_levels / 2 - 1), min=1e-6)

            x_neg = torch.where(x_group <= 0, x_group, torch.zeros_like(x_group))
            x_pos = torch.where(x_group > 0, x_group, torch.zeros_like(x_group))

            x_neg_quant = round_ste(x_neg / delta_l)
            x_pos_quant = round_ste(x_pos / delta_u)

            x_dequant = x_neg_quant * delta_l + x_pos_quant * delta_u
        else:
            delta = torch.clamp((M - m) / (self.n_levels - 1), min=1e-6)
            zero_point = -torch.round(m / delta).detach()
            
            x_int = round_ste(x_group / delta) + zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - zero_point) * delta

        return x_dequant.view_as(x)


class PerTokenLog2Quantizer(nn.Module):
    def __init__(self, n_bits: int = 6, dim: tuple = None, scale_method: tuple = ('max', 0), leaf_param: bool = True):
        super(PerTokenLog2Quantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.dim = (0, 1, 2)

    def forward(self, x: torch.Tensor):
        reduce_dims = [i for i in range(x.ndim) if i not in self.dim]
        delta = x.amax(dim=reduce_dims, keepdim=True).clamp(min=1e-6)

        x_int = round_ste(-torch.log2(x / delta + 1e-20))
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = delta * 2 ** (-x_quant)
        x_dequant[mask] = 0
        
        return x_dequant


class Shift_and_Sum_Quantizer(PerTokenQuantizer):
    def __init__(self, n_bits: int = 8, dim: tuple = None, scale_method: tuple = ('max', 0), leaf_param: bool = True):
        super(Shift_and_Sum_Quantizer, self).__init__(n_bits, dim, scale_method, leaf_param)
        
    def forward(self, x: torch.Tensor, attn: torch.Tensor):
        '''
        x [b, h, s, c]: input value tokens
        attn [b, h, s]: boolean variable which indicates high attention value
        '''
        reduce_dims = [i for i in range(x.ndim) if i not in self.dim]
        M = x.amax(dim=reduce_dims, keepdim=True)
        m = x.amin(dim=reduce_dims, keepdim=True)

        delta = torch.clamp((M - m) / (self.n_levels - 1), min=1e-6)
        zero_point = -torch.round(m / delta).detach()

        # start quantization
        x_int = round_ste(x / delta) + zero_point

        # value multi-quantization
        for log_n_quant in range(1, attn.max().long().item() + 1):
            n_quant = 2 ** log_n_quant

            x_shift = 0
            shift = 1 / (2 * n_quant)
            for i in range(-n_quant // 2, n_quant // 2):
                x_shift += round_ste(x / delta + (2 * i + 1) * shift) / n_quant
            x_shift += zero_point

            x_int = torch.where((attn == log_n_quant).unsqueeze(-1).repeat(1,1,1,x.size(-1)), x_shift, x_int)

        # apply to token positions with high attention scores
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta
        
        return x_dequant