import numpy as np
import torch
from quant.quant_layer import QuantModule, QuantMatMul
from quant.quant_model import QuantVAR
from quant.quant_block import BaseQuantBlock, QSelfAttention, QFFN
from quant.quantizer import UniformAffineQuantizer, GroupAffineQuantizer
from quant.adaptive_rounding import AdaRoundQuantizer, AdaRoundGroupQuantizer
from optim.data_utils import save_inp_oup_data
from typing import Tuple

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum(-1).mean()


def block_reconstruction(model: QuantVAR, block: BaseQuantBlock, labels: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0, p: float = 2.0, lr: float = 4e-5, wwq: bool = True, waq: bool = True,
                         keep_gpu: bool = True, cfg: float=1.5, seed: int=0, idx_Bls: torch.LongTensor = None):
    '''get input and set scale'''
    cached_inps, cached_outs = save_inp_oup_data(model, block, labels, wwq, waq, batch_size, keep_gpu=keep_gpu, cfg=cfg, seed=seed, idx_Bls=idx_Bls)
    '''set state'''
    block.set_quant_state(True, True)

    '''set quantizer'''
    round_mode = 'learned_hard_sigmoid'
    # Replace weight quantizer to AdaRoundQuantizer
    w_para = []
    w_opt = None
    w_scheduler = None

    for module in block.modules():
        '''QuantModule'''
        if isinstance(module, QuantModule):
            if isinstance(module.weight_quantizer, UniformAffineQuantizer):
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode, weight_tensor=module.org_weight.data)
            elif isinstance(module.weight_quantizer, GroupAffineQuantizer):
                module.weight_quantizer = AdaRoundGroupQuantizer(uaq=module.weight_quantizer, round_mode=round_mode, weight_tensor=module.org_weight.data)
            module.weight_quantizer.soft_targets = True
            w_para += [module.weight_quantizer.alpha]
            module.weight_quantizer.delta = module.weight_quantizer.delta.detach().clone()
            module.weight_quantizer.zero_point = module.weight_quantizer.zero_point.detach().clone()

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=lr)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, T_max=iters, eta_min=0.)

    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p)
    device = 'cuda'
    num_res = len(cached_inps[0])
    sz = cached_inps[0][0].size(0)

    for i in range(iters):
        idx = torch.randint(0, sz, (batch_size,))
        w_opt.zero_grad()

        if isinstance(block, QSelfAttention): block.kv_caching(True) # Turn on kv-caching for QSelfAttention

        cur_out, out_quant = [], []
        for res in range(num_res):
            cur_inp = cached_inps[0][res][idx].to(device)
            cur_out.append(cached_outs[res][idx].to(device))
            out_quant.append(block(cur_inp))
        if isinstance(block, QSelfAttention): block.kv_caching(False)

        err = loss_func(out_quant, cur_out)

        err.backward()

        w_opt.step()
        if w_scheduler:
            w_scheduler.step()
    torch.cuda.empty_cache()

    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False


class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = 0
            for pred_, tgt_ in zip(pred, tgt):
                rec_loss += lp_loss(pred_, tgt_, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))