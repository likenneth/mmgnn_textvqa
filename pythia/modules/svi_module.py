import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.layers import ReLUWithWeightNormFC


class SemanticGraphAggregator(nn.Module):
    def __init__(self, im_sem_embed_dim, sem_sem_embed_dim, obj_dim=2048, sem_dim=300):
        super(SemanticGraphAggregator, self).__init__()
        self.im_sem_embed_dim = im_sem_embed_dim
        self.sem_sem_embed_dim = sem_sem_embed_dim

        # for image_semantic_aggregator
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, im_sem_embed_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, im_sem_embed_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(im_sem_embed_dim * 2, sem_dim)

        # for semantic_semantic_aggregator
        self.sem_embed_ss_1 = ReLUWithWeightNormFC(sem_dim, sem_sem_embed_dim)
        self.sem_embed_ss_2 = ReLUWithWeightNormFC(sem_dim, sem_sem_embed_dim)
        self.sem_sem_combined = ReLUWithWeightNormFC(sem_dim * 2, sem_dim)

    def reset_parameters(self):
        pass

    def image_semantic_aggregator(self, i, s):
        # calculate the attention weights
        i_proj = self.im_embed_is(i)  # [batch, obj_num, im_sem_embed_dim]
        s_proj = self.sem_embed_is(s)  # [batch, ocr_num, im_sem_embed_dim]

        # attention matrix
        similarity = torch.matmul(s_proj, i_proj.permute(0, 2, 1))
        att = F.softmax(similarity, dim=2)  # [batch, ocr_num, ocr_num]
        i_att = torch.matmul(att, i_proj)  # [batch, ocr_num, im_sem_embed_dim]

        # aggregate the information, add aligned with aggregated
        combine = self.im_sem_combined(torch.cat((s_proj, i_att), 2))  # [batch, ocr_num, sem_dim]
        return combine

    def semantic_semantic_aggregator(self, s, mask_s):
        # calculate the attention weights
        s_proj_1 = self.sem_embed_ss_1(s)  # [batch, ocr_num, sem_sem_embed_dim]
        s_proj_2 = self.sem_embed_ss_2(s)  # [batch, ocr_num, sem_sem_embed_dim]

        # attention matrix
        similarity = torch.matmul(s_proj_1, s_proj_2.permute(0, 2, 1))
        mask = torch.arange(similarity.size(2))[None, None, :].to(mask_s.device) < mask_s[:, None, None]

        inf_tmp = torch.ones_like(similarity) * float('-inf')
        inf_tmp[mask.expand(-1, similarity.size(1), -1)] = 0
        mask_sim = similarity + inf_tmp
        att = F.softmax(mask_sim, dim=2)  # [batch, ocr_num, ocr_num]
        i_att = torch.matmul(att, s)  # [batch, ocr_num, sem_dim]

        # aggregate the information, add attended with origin
        combine = self.sem_sem_combined(torch.cat((s, i_att), 2))  # [batch, ocr_num, sem_dim]
        return combine

    def forward(self, i, s, mask_s):
        """
        # all below should be batched
        :param i: [100, 2048]
        :param s: [50, 300]
        :param mask_s: int, <num_tokens> <= 50
        :return: updated s with identical shape
        """

        h_is = self.image_semantic_aggregator(i, s)
        h_ss = self.semantic_semantic_aggregator(s, mask_s)
        s_new = s + h_is + h_ss
        # TODO: try different designs
        return s_new


class ValueGraphAggregator(nn.Module):
    def __init__(self, vs_proj_dim, value_dim=300, sem_dim=300):
        super(ValueGraphAggregator, self).__init__()
        self.vs_proj_dim = vs_proj_dim
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, vs_proj_dim)
        self.value_embed_is = ReLUWithWeightNormFC(value_dim, vs_proj_dim)
        self.output_layer = ReLUWithWeightNormFC(vs_proj_dim * 2, value_dim)

    def reset_parameters(self):
        pass

    def forward(self, s, v, mask_s):
        """
        # all below should be batched
        :param s: [50, 300]
        :param v: [50, 300]
        :param mask_s: int, <num_tokens> <= 50
        :return: updated v with identical shapes
        """

        # alignment, attention
        s_proj = self.sem_embed_is(s)  # [batch, ocr_num, vs_proj_dim]
        v_proj = self.sem_embed_is(v)  # [batch, ocr_num, vs_proj_dim]
        similarity = torch.matmul(s_proj, s_proj.transpose(2, 1))  # [batch, ocr_num, ocr_num]
        mask = torch.arange(similarity.size(2))[None, None, :].to(mask_s.device) < mask_s[:, None, None]
        # similarity[~(mask.expand(-1, similarity.size(1), -1))] = float('-inf')
        # att = F.softmax(similarity.clone(), dim=2)  # [batch, ocr_num, ocr_num]
        inf_tmp = torch.ones_like(similarity) * float('-inf')
        inf_tmp[mask.expand(-1, similarity.size(1), -1)] = 0
        mask_sim = similarity + inf_tmp
        att = F.softmax(mask_sim, dim=2)  # [batch, ocr_num, ocr_num]

        # aggregate
        att_v = torch.matmul(att, v_proj)  # [batch, ocr_num, vs_proj_dim]
        v_new = self.output_layer(torch.cat((v_proj, att_v), dim=2))  # [batch, ocr_num, value_dim]

        return v + v_new


class VisualGraphAggregator(nn.Module):
    def __init__(self, is_proj_dim, obj_dim=2048, sem_dim=300):
        super(VisualGraphAggregator, self).__init__()
        self.im_sem_embed_dim = is_proj_dim
        self.im_embed_is = ReLUWithWeightNormFC(obj_dim, is_proj_dim)
        self.sem_embed_is = ReLUWithWeightNormFC(sem_dim, is_proj_dim)
        self.im_sem_combined = ReLUWithWeightNormFC(is_proj_dim * 2, obj_dim)

    def reset_parameters(self):
        pass

    def forward(self, i, s, pi, ps, mask_s):
        """
        :param i: [128, 100, 2048]
        :param s: [128, 50, 300]
        :param pi: [128, 100, 2], [x, y]
        :param ps: [128, 50, 2], same as above
        :param mask_s: [128], <num_tokens> <= 50
        :return: updated i with identical shapes
        """

        # calculate distances
        xi = pi[:, :, 0:1]  # [128, 100, 1]
        xs = ps[:, :, 0:1].transpose(1, 2)  # [128, 1, 50]
        yi = pi[:, :, 1:2]  # [128, 100, 1]
        ys = ps[:, :, 1:2].transpose(1, 2)  # [128, 1, 50]
        distances = torch.sqrt(torch.pow(xi - xs, 2) + torch.pow(yi - ys, 2))  # [batch, obj_num, ocr_num]

        # calculate masked attention
        mask = torch.arange(distances.size(2))[None, None, :].to(mask_s.device) < mask_s[:, None, None]
        # distances[~(mask.expand(-1, distances.size(1), -1))] = float('-inf')
        # att = F.softmax(distances.clone(), dim=2)  # [batch, obj_num, ocr_num]
        inf_tmp = torch.ones_like(distances) * float('-inf')
        inf_tmp[mask.expand(-1, distances.size(1), -1)] = 0
        mask_sim = distances + inf_tmp
        att = F.softmax(mask_sim, dim=2)  # [batch, ocr_num, ocr_num]

        # alignment, attention
        i_proj = self.im_embed_is(i)  # [batch, obj_num, im_sem_embed_dim]
        s_proj = self.sem_embed_is(s)  # [batch, ocr_num, im_sem_embed_dim]
        i_att = torch.matmul(att, s_proj)  # [batch, obj_num, im_sem_embed_dim]

        # aggregate the information, add aligned with aggregated
        i_new = self.im_sem_combined(torch.cat((i_proj, i_att), 2))  # [batch, obj_num, obj_dim]
        return i + i_new


class MultiModalGNN(nn.Module):
    def __init__(self, im_sem_embed_dim, sem_sem_embed_dim, vs_proj_dim, is_proj_dim, net=None, eps=0, train_eps=False):
        super(MultiModalGNN, self).__init__()

        self.im_sem_embed_dim = im_sem_embed_dim
        self.sem_sem_embed_dim = sem_sem_embed_dim
        self.vs_proj_dim = vs_proj_dim
        self.is_proj_dim = is_proj_dim

        self.net = net
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

        self.semantic_graph_aggregator = SemanticGraphAggregator(self.im_sem_embed_dim, self.sem_sem_embed_dim)
        self.visual_graph_aggregator = VisualGraphAggregator(self.vs_proj_dim)
        self.value_graph_aggregator = ValueGraphAggregator(self.is_proj_dim)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.net)

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, i, s, v, pi, ps, mask_v, mask_s):
        """
        # all below should be batched
        :param i: [100, 2048]
        :param s: [50, 300]
        :param v: [50, 300]
        :param pi: [100, 2], [x, y]
        :param ps: [50, 2], same as above
        :param mask_v: int, <num_value> <= <num_tokens> <= 100
        :param mask_s: int, <num_tokens> <= 50
        :return: updated i, s, v with identical shapes
        """

        # Update infos in a designed order: s -> v -> i
        s = self.semantic_graph_aggregator(i, s, mask_s)
        v = self.value_graph_aggregator(s, v, mask_s)
        i = self.visual_graph_aggregator(i, s, pi, ps, mask_s)

        return i, s, v


if __name__ == '__main__':
    _i = torch.randn(128, 100, 2048)
    _s = torch.randn(128, 50, 300)
    _v = torch.randn(128, 50, 300)
    _pi = torch.randn(128, 100, 2)
    _ps = torch.randn(128, 50, 2)
    _mask_i = torch.randint(0, 100, (128,))
    _mask_s = torch.randint(0, 50, (128,))
    module = MultiModalGNN(200, 250, 300, 400)
    for x in module(_i, _s, _v, _pi, _ps, _mask_i, _mask_s):
        print(x.shape)
