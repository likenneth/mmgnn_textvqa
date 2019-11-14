import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from pythia.modules.layers import ReLUWithWeightNormFC, LinearTransform
from geolib.inits import glorot


class SI_GNN(nn.Module):
    def __init__(self, f_engineer, bb_dim, fvd, fsd, l_dim, inter_dim, K, dropout):
        super(SI_GNN, self).__init__()
        self.f_engineer = f_engineer
        self.bb_dim = bb_dim
        self.fvd = fvd
        self.fsd = fsd
        self.l_dim = l_dim
        self.inter_dim = inter_dim
        self.K = K  # attention heads
        self.dropout = dropout

        self.bb_proj = LinearTransform(10, self.bb_dim)
        self.W1 = Parameter(torch.Tensor(self.K, self.bb_dim, self.inter_dim), requires_grad=True)
        self.W2 = Parameter(torch.Tensor(self.K, self.bb_dim, self.inter_dim), requires_grad=True)
        # self.fs_fa1 = LinearTransform(self.fsd + self.bb_dim, self.fvd - self.fsd)
        # self.fs_fa2 = LinearTransform(self.fsd + self.bb_dim, self.fvd - self.fsd)
        # self.fs_fa3 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.fs_fa4 = LinearTransform(self.fsd + self.bb_dim, self.fvd + self.bb_dim)
        self.l_proj2 = LinearTransform(self.l_dim, self.fvd + self.bb_dim)
        self.l_proj3 = LinearTransform(self.l_dim, self.fvd + self.bb_dim)
        # self.fv_fa1 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.fv_fa2 = LinearTransform(self.fvd + self.bb_dim, self.fvd + self.bb_dim)
        self.output_proj1 = ReLUWithWeightNormFC(self.bb_dim + self.fvd, self.fsd)
        self.output_proj2 = ReLUWithWeightNormFC(self.bb_dim + self.fvd, self.fvd)
        self.epsilon = Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W1)
        glorot(self.W2)
        nn.init.normal_(self.epsilon)

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
        :param adj: [B, 50, 100]
        :param mask_s: [B]
        :return: loss induced from this attention, the more average, the more penalty
        """
        fro_norm = torch.sum(torch.norm(adj, dim=-1), dim=-1) / mask_s.to(torch.float)
        return -torch.sum(fro_norm)

    def forward(self, l, s, ps, mask_s, v_ori, pv, mask_v, k_valve=4, it=1, penalty_ratio=10):
        """
        # all below should be batched
        :param l: [2048], to guide edge strengths, by attention
        :param s: [50, 300]
        :param ps: [50, 4], same as above
        :param mask_s: int, <num_tokens> <= 50
        :param v_ori: [loc, vfd]
        :param pv: [100, 4]
        :param mask_v: [1]
        :param k_valve: the k in top_k, to control flow from v to s
        :param it: iterations for GNN
        :param penalty_ratio: the ratio need to be shrunk by penalty loss
        :return: updated i and s with identical shape
        """
        loc = mask_v[0]  # number of image features
        s_bb = self.bb_process(ps)  # [B, 50, bb_dim]
        v_bb = self.bb_process(pv)  # [B, 100, bb_dim]
        # s = torch.cat([s, s_bb], dim=2)  # [B, 50, bb_dim + fsd]
        v = torch.cat([v_ori, v_bb], dim=2)  # [B, 50, bb_dim + fvd]
        l = l.unsqueeze(1)  # [B, 1, l_dim]

        inf_tmp = torch.ones(ps.size(0), 50, loc).to(l.device) * float('-inf')
        mask1 = (torch.arange(50).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).repeat(1, 1, loc)
        inf_tmp[mask1] = 0

        output_mask = (torch.arange(50).to(mask_s.device)[None, :] < mask_s[:, None]).unsqueeze(2).to(s.dtype)

        for _ in range(it):
            s_bb_formul = torch.matmul(s_bb.unsqueeze(1), self.W1.unsqueeze(0))  # [B, K, 50, inter_dim]
            v_bb_formul = torch.matmul(v_bb.unsqueeze(1), self.W2.unsqueeze(0))  # [B, K, 100, inter_dim]
            adj = torch.matmul(s_bb_formul, v_bb_formul.transpose(2, 3))  # [B, K, 50, 100]
            adj = torch.mean(adj, dim=1)  # [B, 50, 100]
            # index_mask = torch.topk(adj, loc - k_valve, dim=-1, largest=False, sorted=False)[-1]
            # adj.scatter_(-1, index_mask, float("-inf"))

            adj = F.softmax(adj, dim=2)  # [B, 50, 100]
            adj = self.cooling(adj, temperature=0.25) * output_mask

            prepared_s_source = self.output_proj2(
                self.fs_fa4(torch.cat([s, s_bb], dim=-1)) * self.l_proj3(l))  # [B, 50, fvd]
            # prepared_s_source = self.output_proj2(self.fs_fa4(torch.cat([s, s_bb], dim=-1)))  # [B, 50, fvd]
            prepared_s_source = F.dropout(prepared_s_source, self.dropout)

            prepared_v_source = self.output_proj1(self.fv_fa2(v) * F.softmax(self.l_proj2(l), dim=-1))  # [B, 100, fsd]
            # prepared_v_source = self.output_proj1(self.fv_fa2(v))  # [B, 100, fsd]

            prepared_v_source = F.dropout(prepared_v_source, self.dropout)

            new_ele = torch.matmul(adj.transpose(1, 2), prepared_s_source)
            v = self.epsilon * new_ele + v_ori  # [B, loc, fvd]
            s = torch.cat([s, torch.matmul(adj, prepared_v_source)], dim=2)  # [B, 50, 2 * fsd]
        return s * output_mask, v, adj + inf_tmp, self.att_loss(adj, mask_s) / penalty_ratio

    def cooling(self, adj, temperature=0.5):
        """
        :param adj: [B, 50, 50], with adj value in 0 to 1, usually after softmax
        :return: cooled adj of the same shape
        """
        if self.training:
            adj = adj + (torch.randn(adj.shape) / 592).to(adj.device)
        adj = torch.pow(F.relu(adj), 1 / temperature)
        adj = adj / torch.sum(adj, dim=-1, keepdim=True)
        return adj
