import os
import pickle
import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.sparse_graph_model import Model


@registry.register_model("sgm_mmgnn")
class LoRRA(Pythia):
    def __init__(self, config):
        super().__init__(config)
        self.clk = 0
        self.f_engineer = config.f_engineer
        self.feature_dim = config.gnn.feature_dim
        self.hid_dim = config.gnn.hid_dim
        self.dropout = config.gnn.dropout
        self.l_dim = config.gnn.l_dim
        self.out_dim = config.gnn.out_dim
        self.n_kernels = config.gnn.n_kernels
        self.nei_size = config.gnn.nei_size
        self.image_feature_embeddings_out_dim = self.out_dim

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        self.s_gnn = Model(self.l_dim, self.feature_dim, self.hid_dim, self.out_dim, self.dropout,
                           n_kernels=self.n_kernels, neighbourhood_size=self.nei_size)
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
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

    def record_for_analysis(self, id, adj):
        with open(os.path.join("/home/like/Workplace/textvqa/save/error_analysis/gnn_att",
                               self.config.model + "_" + self.config.code_name + "_" + str(self.clk) + ".p"),
                  'wb') as f:
            pickle.dump({"question_id": id, "att": adj}, f, protocol=-1)

    def forward(self, sample_list):
        self.clk += 1
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total = self.process_text_embedding(sample_list)

        i0 = sample_list["image_feature_0"]
        i1 = sample_list["image_feature_1"]
        s = sample_list["context_feature_0"]
        bb_rcnn = self.f_process(sample_list["bb_rcnn"], sample_list["image_info_0"]["image_w"],
                                 sample_list["image_info_0"]["image_h"], self.f_engineer)
        bb_ocr = self.f_process(sample_list["bb_ocr"], sample_list["image_info_0"]["image_w"],
                                sample_list["image_info_0"]["image_h"], self.f_engineer)

        i0 = self.image_feature_encoders[0](i0)

        image_embedding_total, gnn_adj = self.s_gnn(text_embedding_total, torch.cat([i0[:, :100], bb_rcnn], dim=2), 100)
        # i0 = torch.cat([i0, torch.zeros(i0.size(0), 37, i0.size(2)).to(i0.device).to(i0.dtype)], dim=1)
        #
        # image_embedding_total, _ = self.process_feature_embedding("image", sample_list, text_embedding_total,
        #                                                           image_f=[i0])

        context_embedding_total, _ = self.process_feature_embedding("context", sample_list, text_embedding_total,
                                                                    ["order_vectors"], context_f=[s])

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(["image", "text"], [image_embedding_total, text_embedding_total,
                                                                      context_embedding_total], )

        scores = self.calculate_logits(joint_embedding)

        if self.clk % 500 == 0:
            self.record_for_analysis(sample_list["question_id"], gnn_adj)

        return {"scores": scores, "att": gnn_adj}
