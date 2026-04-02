import torch
import torch.nn.functional as F
from quant.quant_layer import QuantModule, Union
from quant.quant_model import QuantVAR
from quant.quant_block import BaseQuantBlock
from typing import Tuple


def save_inp_oup_data(model: QuantVAR, layer: Union[QuantModule, BaseQuantBlock], labels: torch.Tensor,
                      wq: bool = False, aq: bool = False, batch_size: int = 32, keep_gpu: bool = True,
                      cfg: float = 1.5, seed: int = 0, multi_scale: bool = True, idx_Bls: torch.LongTensor = None):
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, wq=wq, aq=aq, cfg=cfg, seed=seed)
    cached_inps, cached_outs = None, None

    if multi_scale:
        for label_B, idx_Bl in zip(labels, idx_Bls):
            cur_inp_, cur_out_ = get_inp_out(label_B, idx_Bl)
            if cached_inps is None: cached_inps, cached_outs = [[] for _ in range(len(cur_inp_))], [[] for _ in range(len(cur_out_))]
            for i, (cur_inp, cur_out) in enumerate(zip(cur_inp_, cur_out_)):
                cached_inps[i].append(cur_inp.cpu())
                cached_outs[i].append(cur_out.cpu())
    else:
        for label_B, idx_Bl in zip(labels, idx_Bls):
            cur_inp_, cur_out_ = get_inp_out(label_B, idx_Bl)
            if cached_inps is None: cached_inps, cached_outs = [[]], [[]]
            cached_inps[0].append(cur_inp_[0].cpu())
            cached_outs[0].append(cur_out_[0].cpu())

    num_res = len(cached_inps)
    for i in range(num_res):
        cached_inps[i] = torch.cat([x for x in cached_inps[i]], dim=0)
        cached_outs[i] = torch.cat([x for x in cached_outs[i]], dim=0)
    torch.cuda.empty_cache()

    if keep_gpu:
        for i in range(num_res):
            cached_inps[i] = cached_inps[i].to(device)
            cached_outs[i] = cached_outs[i].to(device)
            
    return (cached_inps,), cached_outs


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = []
        self.output_store = []

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store.append(input_batch[0].detach().clone())
        if self.store_output:
            self.output_store.append(output_batch.detach().clone())
        if self.stop_forward:
            raise StopForwardException

    def clear(self):
        self.input_store = []
        self.output_store = []


class GetLayerLogit:
    def __init__(self, model: QuantVAR, device: torch.device, cfg: float = 1.5, seed: int = 0):
        self.model = model
        self.device = device

        self.cfg = cfg
        self.seed = seed

        self.patch_nums = self.model.patch_nums
        self.rng = self.model.rng

        self.logits = [[] for _ in range(len(self.patch_nums))]

    def __call__(self, label_B, idx_Bl):
        self.model.set_quant_state(False, False)
        with torch.no_grad():
            self.model.get_rng_state()
            logits_BlV = self.model.autoregressive_infer_cfg(B=label_B.size(0), label_B=label_B.to(self.device), cfg=self.cfg, top_k=900, top_p=0.96, g_seed=self.seed + label_B.item(), fixed_idx_Bl=idx_Bl, more_smooth=False)
            self.model.set_rng_state()

        return logits_BlV


class GetLayerInpOut:
    def __init__(self, model: QuantVAR, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, wq: bool = False, aq: bool = False,
                 cfg: float = 1.5, seed: int = 0):
        self.model = model
        self.layer = layer
        self.device = device
        self.wq = wq
        self.aq = aq
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)

        self.cfg = cfg
        self.seed = seed

        self.patch_nums = self.model.patch_nums
        self.rng = self.model.rng

    def __call__(self, label_B, idx_Bl):
        self.model.set_quant_state(False, False)
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            self.model.get_rng_state()
            _ = self.model.autoregressive_infer_cfg(B=label_B.size(0), label_B=label_B.to(self.device), cfg=self.cfg, top_k=900, top_p=0.96, g_seed=self.seed + label_B.item(), fixed_idx_Bl=idx_Bl, more_smooth=False)
            self.model.set_rng_state()
            input_quant = self.data_saver.input_store
            output_fp = self.data_saver.output_store
            self.data_saver.clear()
            
        handle.remove()

        return input_quant, output_fp