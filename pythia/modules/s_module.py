import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC


class S_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, feature_dim):
        super(S_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim

        self.bb_proj = ReLUWithWeightNormFC(self.f_engineer, self.bb_dim)
        self.s_proj = ReLUWithWeightNormFC(300, self.feature_dim)
        # self.l_proj = ReLUWithWeightNormFC(2048, self.feature_dim + self.bb_dim)
        # self.gate = nn.Tanh()
        self.s_recover = ReLUWithWeightNormFC(self.feature_dim, 300)

    def reset_parameters(self):
        pass

    def forward(self, l, s, ps, mask_s, it=1):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param it: iterations for GNN
        :return: updated s with identical shape
        """

        bb = self.bb_proj(ps)  # [B, 50, bb_dim]
        s_fa = self.s_proj(s)  # [B, 50, feature_dim]
        # l_fa = F.softmax(self.l_proj(l), dim=1)  # [B, feature_dim + bb_dim]

        inf_tmp = torch.ones(bb.size(0), 50, 50).to(l.device) * float('-inf')
        mask1 = torch.max(torch.arange(50)[None, :], torch.arange(50)[:, None])
        mask1 = mask1[None, :, :].to(mask_s.device) < mask_s[:, None, None]
        mask2 = torch.arange(50).unsqueeze(1).expand(-1, 50).to(mask_s.device)[None, :, :] >= mask_s[:, None, None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0
        inf_tmp[torch.eye(50).byte().unsqueeze(0).repeat(bb.size(0), 1, 1)] = float('-inf')
        mask3 = mask_s == 1
        inf_tmp[:, 0, 0][mask3] = 0

        output_mask = (torch.arange(50).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).expand(-1, -1, 300)

        for _ in range(it):
            combined_fea = torch.cat([bb, s_fa], dim=2)  # [B, 50, bb_dim + feature_dim]
            adj = torch.matmul(combined_fea, combined_fea.transpose(1, 2))  # [B, 50, 50]
            # adj = F.softmax(adj / torch.Tensor([combined_fea.size(2)]).to(adj.dtype).to(adj.device).sqrt_() + inf_tmp,
            #                 dim=2)  # [B, 50, 50]
            adj = F.softmax(adj + inf_tmp, dim=2)  # [B, 50, 50]
            s_fa = torch.matmul(adj, s_fa) + s_fa  # [B, 50, feature_dim]

        s_new = self.s_recover(s_fa)

        return s_new * output_mask.to(s_new.dtype)


if __name__ == '__main__':
    from torchviz import make_dot

    _i = torch.randn(128, 100, 2048)
    _s = torch.randn(128, 50, 300)
    _pi = torch.randn(128, 100, 4)
    _ps = torch.randn(128, 50, 4)
    _mask_s = torch.randint(0, 50, (128,))
    _it = 2
    module = S_GNN(200, 200)
    result = module(_i, _s, _pi, _ps, _mask_s, _it)
    for res in result:
        print(res.shape)
    make_dot(result, params=dict(module.named_parameters()))
