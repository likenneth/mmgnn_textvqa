import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC


class SI_GNN(nn.Module):
    def __init__(self, bb_dim, feature_dim):
        super(SI_GNN, self).__init__()
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim

        self.bb_proj = ReLUWithWeightNormFC(4, self.bb_dim)
        self.i_proj = ReLUWithWeightNormFC(2048, self.feature_dim)
        self.s_proj = ReLUWithWeightNormFC(300, self.feature_dim)
        self.l_proj = ReLUWithWeightNormFC(2048, self.feature_dim + self.bb_dim)
        self.gate = nn.Tanh()
        self.i_recover = ReLUWithWeightNormFC(self.feature_dim, 2048)
        self.s_recover = ReLUWithWeightNormFC(self.feature_dim, 300)

    def reset_parameters(self):
        pass

    def forward(self, l, i, s, pi, ps, mask_s, it=1):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param i: [loc, 2048]
        :param s: [50, 300]
        :param pi: [loc, 4], [left, down, right, upper]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param it: iterations for GNN
        :return: updated i and s with identical shape
        """
        self.loc = pi.size(1)
        bb = self.bb_proj(torch.cat((pi, ps), 1))  # [B, loc + 50, bb_dim]
        i_fa = self.i_proj(i)  # [B, loc, feature_dim]
        s_fa = self.s_proj(s)  # [B, 50, feature_dim]
        l_fa = F.softmax(self.l_proj(l), dim=1)  # [B, feature_dim + bb_dim]
        inf_tmp = torch.ones(bb.size(0), self.loc + 50, self.loc + 50).to(i_fa.device) * float('-inf')
        mask1 = torch.max(torch.arange(self.loc + 50)[None, :], torch.arange(self.loc + 50)[:, None])
        mask1 = mask1[None, :, :].to(mask_s.device) < mask_s[:, None, None]
        mask2 = torch.arange(self.loc + 50).unsqueeze(1).expand(-1, self.loc + 50).to(mask_s.device)[None, :,
                :] >= mask_s[:, None, None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0

        for _ in range(it):
            fea = torch.cat((i_fa, s_fa), 1)  # [B, loc + 50, feature_dim]
            edges = torch.cat((fea, bb), 2)  # [B, loc + 50, feature_dim+ bb_dim]
            att = torch.matmul(edges * (l_fa.unsqueeze(1).expand(-1, self.loc + 50, -1)),
                               edges.transpose(1, 2))  # [B, loc + 50, loc + 50]
            att = F.softmax(att + inf_tmp, dim=2)  # [B, loc + 50, loc + 50]
            fea = fea + self.gate(torch.matmul(att, fea))  # [B, loc + 50, feature_dim]

        i_new = self.i_recover(fea[:, :self.loc].clone())
        s_new = self.s_recover(fea[:, self.loc:].clone())

        return i_new, s_new


if __name__ == '__main__':
    from torchviz import make_dot

    _i = torch.randn(128, 100, 2048)
    _s = torch.randn(128, 50, 300)
    _pi = torch.randn(128, 100, 4)
    _ps = torch.randn(128, 50, 4)
    _mask_s = torch.randint(0, 50, (128,))
    _it = 2
    module = SI_GNN(200, 200)
    result = module(_i, _s, _pi, _ps, _mask_s, _it)
    for res in result:
        print(res.shape)
    make_dot(result, params=dict(module.named_parameters()))
