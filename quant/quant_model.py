import torch
import torch.nn as nn

from quant.utils import MatMul
from quant.quant_layer import QuantModule, QuantMatMul
from quant.quant_block import block_specials, BaseQuantBlock
from models.basic_var import AdaLNSelfAttn


class QuantVAR(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        super().__init__()
        self.model = model
        self.patch_nums = self.model.patch_nums
        self.rng = self.model.rng

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, matmul_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, matmul_params: dict = {}):
        specials = block_specials
        for name, child_module in module.named_children():
            if isinstance(child_module, AdaLNSelfAttn):
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params, matmul_params))

            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, matmul_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantMatMul):
                m.set_quant_state(act_quant, act_quant)

    def get_rng_state(self):
        self.model.get_rng_state()
    
    def set_rng_state(self):
        self.model.set_rng_state()

    def forward(self, *input):
        return self.model(*input)

    def autoregressive_infer_cfg(self, **input):
        return self.model.autoregressive_infer_cfg(**input)