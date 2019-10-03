import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

from pythia.modules.layers import ReLUWithWeightNormFC, LinearTransform


class Hard_Output(Module):
    def __init__(self, l_dim, i_dim, s_dim, inter_dim, vocab_size):
        super(Hard_Output, self).__init__()
        self.vocab_size = vocab_size
        self.inter_dim = inter_dim
        self.s_dim = s_dim
        self.i_dim = i_dim
        self.l_dim = l_dim

        self.i_proj = ReLUWithWeightNormFC(self.i_dim, self.inter_dim)
        self.s_proj = ReLUWithWeightNormFC(self.s_dim, self.inter_dim)
        self.l_proj1 = ReLUWithWeightNormFC(self.l_dim, self.inter_dim)
        self.l_proj2 = ReLUWithWeightNormFC(self.l_dim, self.inter_dim)
        self.l_proj3 = ReLUWithWeightNormFC(self.l_dim, self.inter_dim)
        self.si2l = ReLUWithWeightNormFC(2 * self.inter_dim, self.inter_dim)
        self.bias = ReLUWithWeightNormFC(2 * self.inter_dim, 1)
        self.vocab_predict = LinearTransform(2 * self.inter_dim, self.vocab_size)

    def forward(self, l, i, s, copy, mask_copy):
        """
        :param l: [B, l_dim], now is 2048
        :param i: [B, i_dim], noew is 5k
        :param s: [B, s_dim], now is 5k
        :param copy: [B, 50], already-masked
        :param mask_copy: [B]
        :return: scores: [B, 4097 + 50]
        :return: b2s: [B]
        """
        i_fa = self.i_proj(i)
        l_fa1 = self.l_proj1(l)
        l_fa2 = self.l_proj2(l)
        l_fa3 = self.l_proj3(l)
        s_fa = self.s_proj(s)
        si_fa = self.si2l(torch.cat([s_fa, i_fa], dim=-1))
        vocab_res = self.vocab_predict(torch.cat([l_fa2 * s_fa, l_fa3 * i_fa], dim=-1))

        # vocab_output -inf <--- b2s ---> +inf copy mechanism
        b2s = self.bias(torch.cat([l_fa1, si_fa], dim=-1))
        w1 = torch.ones_like(vocab_res, dtype=vocab_res.dtype, device=vocab_res.device, requires_grad=True) * b2s
        w2 = torch.ones_like(copy, dtype=copy.dtype, device=copy.device, requires_grad=True) * b2s

        mask = torch.arange(50).to(copy.device)[None, :] < mask_copy[:, None]
        mask_tmp = torch.zeros_like(copy, dtype=copy.dtype, device=copy.device)
        mask_tmp[mask] = 1
        big_minus_tmp = torch.ones_like(copy, dtype=copy.dtype, device=copy.device) * (-1e20)
        big_minus_tmp[mask] = 0

        final_res = torch.cat([vocab_res - w1, (copy + w2) * mask_tmp + big_minus_tmp], dim=-1)

        return final_res, b2s
