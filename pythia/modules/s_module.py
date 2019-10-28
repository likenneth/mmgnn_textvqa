import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC, LinearTransform


class S_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, feature_dim, l_dim, inter_dim, dropout):
        super(S_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.feature_dim = feature_dim
        self.l_dim = l_dim
        self.inter_dim = inter_dim
        self.dropout = dropout

        self.bb_proj = ReLUWithWeightNormFC(10, self.bb_dim)
        self.fea_fa1 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa2 = ReLUWithWeightNormFC(self.bb_dim + self.feature_dim, self.bb_dim + self.feature_dim)
        self.fea_fa3 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa4 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.fea_fa5 = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), 2 * (self.bb_dim + self.feature_dim))
        self.l_proj1 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.l_proj2 = ReLUWithWeightNormFC(self.l_dim, 2 * (self.bb_dim + self.feature_dim))
        self.output_proj = ReLUWithWeightNormFC(2 * (self.bb_dim + self.feature_dim), self.feature_dim)

    def reset_parameters(self):
        pass

    def bb_process(self, bb):
        """
        :param bb: [B, num, 4], left, down, upper, right
        :return: [B, num(50 or 100), bb_dim]
        """
        bb_size = (bb[:, :, 2:] - bb[:, :, :2])  # 2
        bb_centre = bb[:, :, :2] + 0.5 * bb_size  # 2
        bb_area = (bb_size[:, :, 0] * bb_size[:, :, 1]).unsqueeze(2)  # 1
        bb_shape = (bb_size[:, :, 0] / (bb_size[:, :, 1] + 1e-14)).unsqueeze(2)  # 1
        return self.bb_proj(torch.cat([bb, bb_size, bb_centre, bb_area, bb_shape], dim=-1))

    def att_loss(self, adj, mask_s):
        """
        :param adj: [B, 50, 50]
        :param mask_s: [B]
        :return: loss induced from this attention, the more average, the more penalty
        """
        fro_norm = torch.sum(torch.norm(adj, dim=-1), dim=-1) / mask_s.to(torch.float)
        return -torch.sum(fro_norm)

    def forward(self, l, s, ps, mask_s, it=1, penalty_ratio=10):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param it: iterations for GNN
        :param penalty_ratio: need tobe shrunk by att loss
        :return: updated s with identical shape
        """
        bb = self.bb_process(ps)  # [B, 50, bb_dim]
        s_with_bb = torch.cat([s, bb], dim=2)  # [B, 50, bb_dim + feature_dim]
        l = l.unsqueeze(1).repeat(1, 50, 1)  # [B,50, l_dim]

        inf_tmp = torch.ones(bb.size(0), 50, 50).to(l.device) * float('-inf')
        mask1 = torch.max(torch.arange(50)[None, :], torch.arange(50)[:, None])
        mask1 = mask1[None, :, :].to(mask_s.device) < mask_s[:, None, None]
        mask2 = torch.arange(50).unsqueeze(1).expand(-1, 50).to(mask_s.device)[None, :, :] >= mask_s[:, None, None]
        inf_tmp[mask1] = 0
        inf_tmp[mask2] = 0
        inf_tmp[torch.eye(50).byte().unsqueeze(0).repeat(bb.size(0), 1, 1)] = float('-inf')
        mask3 = mask_s == 1
        inf_tmp[:, 0, 0][mask3] = 0

        output_mask = (torch.arange(50).to(mask_s.device)[None, :] < \
                       mask_s[:, None]).unsqueeze(2).to(s.dtype)

        for _ in range(it):
            combined_fea = torch.cat(
                [s_with_bb, F.dropout(self.fea_fa1(s_with_bb) * self.fea_fa2(s_with_bb), self.dropout)],
                dim=2)  # [B, 50, 2*(bb_dim + feature_dim)]
            l_masked_source = self.fea_fa3(combined_fea) * self.l_proj1(l)  # [B, 50, 2*(bb_dim + feature_dim)]

            l_masked_source = F.dropout(l_masked_source, self.dropout)
            fea_fa4 = F.dropout(self.fea_fa4(combined_fea), self.dropout)
            adj = torch.matmul(fea_fa4, l_masked_source.transpose(1, 2))  # [B, 50, 50]
            adj = F.softmax(adj + inf_tmp, dim=2)  # [B, 50, 50]
            prepared_source = self.fea_fa5(combined_fea) * self.l_proj2(l)  # [B, 50, 2*(bb_dim + feature_dim)]
            messages = self.output_proj(torch.matmul(adj, prepared_source))  # [B, 50, feature_dim]
            s = torch.cat([s, messages], dim=2)  # [B, 50, 2 * feature_dim]

        return s * output_mask, adj * output_mask, self.att_loss(adj * output_mask, mask_s) / penalty_ratio
