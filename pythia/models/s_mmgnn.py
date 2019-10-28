import os
import pickle
import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer
from pythia.modules.si_module import SI_GNN
from pythia.modules.s_module import S_GNN
from pythia.modules.hard_output import Hard_Output


@registry.register_model("s_mmgnn")
class LoRRA(Pythia):
    def __init__(self, config):
        super().__init__(config)
        self.clk = 0
        self.f_engineer = config.f_engineer
        self.si_k_valve = config.si_gnn.k_valve
        self.si_it = config.si_gnn.iteration
        self.s_it = config.s_gnn.iteration
        self.si_penalty = config.si_gnn.penalty
        self.s_penalty = config.s_gnn.penalty
        self.si_inter_dim = config.si_gnn.inter_dim
        self.s_inter_dim = config.s_gnn.inter_dim
        self.K = config.si_gnn.K
        self.output_inter_dim = config.output.inter_dim

        self.bb_dim = config.bb_dim
        self.fsd = config.fsd
        self.fvd = config.fvd
        self.l_dim = config.l_dim
        self.dropout = config.dropout

    def build(self):
        self._init_text_embeddings("text")
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        self.si_gnn = SI_GNN(self.f_engineer, self.bb_dim, self.fvd, self.fsd, self.l_dim, self.si_inter_dim, self.K,
                             self.dropout)
        self.s_gnn = S_GNN(self.f_engineer, self.bb_dim, 2 * self.fsd, self.l_dim, self.s_inter_dim, self.dropout)
        self.output = Hard_Output(self.l_dim, self.fvd, 4 * self.fsd, self.output_inter_dim, 3997)
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
            {"params": self.s_gnn.parameters()},
            {"params": self.si_gnn.parameters()},
            {"params": self.output.parameters()},
        ]

        return params

    def _get_classifier_input_dim(self):
        # Now, the classifier's input will be cat of image and context based features
        return 2 * super()._get_classifier_input_dim()

    def f_process(self, bb, w, h, service):
        """
        :param bb: tensor, [B, 50, 4], left, down, upper, right
        :param w: list, [B]
        :param h: list, [B]
        :param service: the number of features wanted in config
        :return: [B, 50, service]
        """
        # relative_w = (bb[:, :, 2] - bb[:, :, 0]) / torch.Tensor(w).to(bb.device).unsqueeze(1).repeat(1, 50) * 2 - 1
        # relative_h = (bb[:, :, 3] - bb[:, :, 1]) / torch.Tensor(h).to(bb.device).unsqueeze(1).repeat(1, 50) * 2 - 1
        # relative_cp_x = (bb[:, :, 2] + bb[:, :, 0]) / torch.Tensor(w).to(bb.device).unsqueeze(1).repeat(1, 50) - 1
        # relative_cp_y = (bb[:, :, 3] + bb[:, :, 1]) / torch.Tensor(w).to(bb.device).unsqueeze(1).repeat(1, 50) - 1
        K = bb.size(1)
        relative_l = bb[:, :, 0] / torch.Tensor(w).to(bb.device).unsqueeze(1).repeat(1, K)
        relative_d = bb[:, :, 1] / torch.Tensor(h).to(bb.device).unsqueeze(1).repeat(1, K)
        relative_r = bb[:, :, 2] / torch.Tensor(w).to(bb.device).unsqueeze(1).repeat(1, K)
        relative_u = bb[:, :, 3] / torch.Tensor(h).to(bb.device).unsqueeze(1).repeat(1, K)

        if service == 4:
            res = torch.stack([relative_l, relative_d, relative_r, relative_u], dim=2)
            return res

    def record_for_analysis(self, id, si=None, s=None, c=None, b=None):
        with open(os.path.join("/home/like/Workplace/textvqa/save/error_analysis/gnn_att",
                               self.config.model + "_" + self.config.code_name + "_" + str(self.clk) + ".p"),
                  'wb') as f:
            res = {"question_id": id}
            if si is not None:
                res["si_adj"] = si.detach().cpu()
            if s is not None:
                res["s_adj"] = s.detach().cpu()
            if c is not None:
                res["c_adj"] = c.detach().cpu()
            if b is not None:
                res["b2s"] = b.detach().cpu()
            pickle.dump(res, f, protocol=-1)

    def forward(self, sample_list):
        self.clk += 1
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        i0 = sample_list["image_feature_0"]
        i1 = sample_list["image_feature_1"]
        s = sample_list["context_feature_0"]
        bb_ocr = self.f_process(sample_list["bb_ocr"], sample_list["image_info_0"]["image_w"],
                                sample_list["image_info_0"]["image_h"], self.f_engineer)
        bb_rcnn = self.f_process(sample_list["bb_rcnn"], sample_list["image_info_0"]["image_w"],
                                 sample_list["image_info_0"]["image_h"], self.f_engineer)
        bb_resnet = self.f_process(sample_list["bb_resnet"], sample_list["image_info_0"]["image_w"],
                                   sample_list["image_info_0"]["image_h"], self.f_engineer)

        i0 = self.image_feature_encoders[0](i0)
        s, i, si_adj, loss1 = self.si_gnn(text_embedding_total, s, bb_ocr,
                                          sample_list["context_info_0"]["max_features"].clamp_(min=1, max=50),
                                          torch.cat([i0[:, :100], i1], dim=1),
                                          torch.cat([bb_rcnn, bb_resnet], dim=1),
                                          sample_list["image_info_0"]["max_features"] + 196,
                                          k_valve=self.si_k_valve,
                                          it=self.si_it, penalty_ratio=self.si_penalty)  # [B, 50, 600]
        image_embedding_total, _, _ = self.process_feature_embedding("image", sample_list, text_embedding_total,
                                                                     image_f=[i[:, :100]])

        s, gnn_adj, loss2 = self.s_gnn(text_embedding_total, s, bb_ocr, sample_list["context_info_0"]["max_features"],
                                       self.s_it, self.s_penalty)
        context_embedding_total, combine_att, raw_att = self.process_feature_embedding("context", sample_list,
                                                                                       text_embedding_total,
                                                                                       context_f=[s])

        scores, bias_towards_context = self.output(text_embedding_total, image_embedding_total, context_embedding_total,
                                                   raw_att.squeeze(2),
                                                   sample_list["context_info_0"]["max_features"])

        return {"scores": scores, "loss1": loss1, "loss2": loss2,
                "att": {"si_att": si_adj, "s_att": gnn_adj, "combine_att": combine_att, "b2s": bias_towards_context}}
