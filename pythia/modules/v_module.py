import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC


class V_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, feature_dim, l_dim, inter_dim):
        super(V_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim
        self.l_dim = l_dim
        self.inter_dim = inter_dim

        self.bb_proj = ReLUWithWeightNormFC(self.f_engineer, self.bb_dim)
        self.inter_proj = ReLUWithWeightNormFC(self.feature_dim + self.bb_dim + self.l_dim, self.inter_dim)
        # self.l_proj = ReLUWithWeightNormFC(2048, self.feature_dim + self.bb_dim)
        # self.gate = nn.Tanh()
        # self.v_recover = ReLUWithWeightNormFC(self.feature_dim, 2048)

    def reset_parameters(self):
        pass

    def forward(self, l, v, pv, mask_v, it=1):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param v: [100, 2048]
        :param pv: [100, 4], same as above
        :param mask_v: int, <num_tokens> <= 100
        :param it: iterations for GNN
        :return: updated s with identical shape
        """
        bb = self.bb_proj(pv)  # [B, 100, bb_dim]
        l = l.unsqueeze(1).repeat(1, 100, 1)  # [B, 100, 2048]
        # l_fa = F.softmax(self.l_proj(l), dim=1)  # [B, feature_dim + bb_dim]

        inf_tmp = torch.ones(bb.size(0), 100, 100).to(l.device) * float('-inf')
        mask1 = torch.max(torch.arange(100)[None, :], torch.arange(100)[:, None])
        mask1 = mask1[None, :, :].to(mask_v.device) < mask_v[:, None, None]
        mask2 = torch.arange(100).unsqueeze(1).expand(-1, 100).to(mask_v.device)[None, :, :] >= mask_v[:, None,
                                                                                                None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0
        inf_tmp[torch.eye(100).byte().unsqueeze(0).repeat(bb.size(0), 1, 1)] = float('-inf')

        output_mask = (torch.arange(100).to(mask_v.device)[None, :] < mask_v[:, None]).unsqueeze(2).expand(-1, -1,
                                                                                                           2048)

        for _ in range(it):
            combined_fea = torch.cat([v, bb, l], dim=2)  # [B, 100, bb_dim + feature_dim + l_dim]
            condensed = self.inter_proj(combined_fea)  # [B, 100, inter_dim]
            adj = torch.matmul(condensed, condensed.transpose(1, 2))  # [B, 100, 100]
            # adj = F.softmax(adj / torch.Tensor([condensed.size(2)]).to(adj.dtype).to(adj.device).sqrt_() + inf_tmp,
            #                 dim=2)  # [B, 100, 100]
            adj = F.softmax(adj + inf_tmp, dim=2)  # [B, 100, 100]
            v = torch.matmul(adj, v) + v  # [B, 100, feature_dim]

        # v_new = self.v_recover(v_fa)

        return v * output_mask.to(v.dtype), adj


if __name__ == '__main__':
    from torchviz import make_dot

    _i = torch.randn(128, 100, 2048)
    _s = torch.randn(128, 100, 2048)
    _pi = torch.randn(128, 100, 4)
    _ps = torch.randn(128, 100, 4)
    _mask_s = torch.randint(0, 100, (128,))
    _it = 2
    module = V_GNN(200, 200)
    result = module(_i, _s, _pi, _ps, _mask_s, _it)
    for res in result:
        print(res.shape)
    make_dot(result, params=dict(module.named_parameters()))
