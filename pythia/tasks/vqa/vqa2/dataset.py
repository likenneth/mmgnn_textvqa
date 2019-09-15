# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle

import torch
import tqdm
import numpy as np

from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.tasks.image_database import ImageDatabase
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.general import get_pythia_root


class VQA2Dataset(BaseDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("vqa2", dataset_type, config)
        imdb_files = self.config.imdb_files

        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.imdb = ImageDatabase(self.imdb_file)

        self.kwargs = kwargs
        self.image_depth_first = self.config.image_depth_first
        self._should_fast_read = self.config.fast_read

        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

        self._use_features = False
        if hasattr(self.config, "image_features"):
            self._use_features = True
            self.features_max_len = self.config.features_max_len
            self._return_info = self.config.get("return_info", True)

            all_image_feature_dirs = self.config.image_features[dataset_type]
            curr_image_features_dir = all_image_feature_dirs[imdb_file_index]
            curr_image_features_dir = curr_image_features_dir.split(",")
            curr_image_features_dir = self._get_absolute_path(curr_image_features_dir)

            self.features_db = FeaturesDataset(
                "coco",
                directories=curr_image_features_dir,
                depth_first=self.image_depth_first,
                max_features=self.features_max_len,
                fast_read=self._should_fast_read,
                imdb=self.imdb,
                return_info=self._return_info,
            )

        self.fast_dir = os.path.join(config.fast_dir, self._dataset_type)
        self.fasted = set()
        if not os.path.exists(self.fast_dir):
            os.mkdir(self.fast_dir)

        for sample in os.listdir(self.fast_dir):
            self.fasted.add(int(sample[:-2]))

        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def __len__(self):
        return len(self.imdb)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self._name, self._dataset_type
                )
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.imdb)), miniters=100, disable=not is_main_process()
            ):
                self.cache[idx] = self.load_item(idx)

    def get_item(self, idx):
        if idx in self.fasted:
            with open(os.path.join(self.fast_dir, str(idx) + ".p"), 'rb') as f:
                fd = pickle.load(f)
            return fd
        else:
            sd = self.load_item(idx)
            # rico: "target" is also included as saved

            with open(os.path.join(self.fast_dir, str(idx) + ".p"), 'wb') as f:
                pickle.dump(sd, f, protocol=-1)
            self.fasted.add(idx)
            return sd
        # return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if "question_tokens" in sample_info:
            text_processor_argument = {"tokens": sample_info["question_tokens"]}
        else:
            text_processor_argument = {"text": sample_info["question"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
        )

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)

        # add bounding box features
        current_sample = self.add_center_points_info(sample_info, current_sample)
        current_sample = self.add_all_bb_info(sample_info, current_sample)

        # reorder rcnn features according to confidences
        current_sample = self.re_order(current_sample)

        return current_sample

    def add_center_points_info(self, sample_info, sample):
        # 1. add center points of bounding boxes info of faster-rcnn features
        bbox_rcnn = torch.from_numpy(sample["image_info_0"]["boxes"])  # [100, 4]
        sample["center_point_rcnn"] = torch.stack(
            ((bbox_rcnn[:, 0] + bbox_rcnn[:, 2]) / 2, (bbox_rcnn[:, 1] + bbox_rcnn[:, 3]) / 2), dim=1)

        # 2. add center points of ResNet spatial features
        # generate coordinates of the grids in feature map
        grid_num = 14
        x_start = sample_info['image_width'] / grid_num / 2
        y_start = sample_info['image_height'] / grid_num / 2
        x_end = sample_info['image_width'] - x_start
        y_end = sample_info['image_height'] - y_start
        x = self.floatrange(x_start, x_end, grid_num)
        y = self.floatrange(y_start, y_end, grid_num)
        x_grid, y_grid = np.meshgrid(x, y)
        x_grid = x_grid.reshape([-1])
        y_grid = y_grid.reshape([-1])
        sample['center_point_resnet'] = torch.stack(tuple(torch.tensor([x_grid[i], y_grid[i]])
                                                          for i in range(grid_num ** 2)))  # [196, 2]

        # 3. add center points of OCR
        bbox_ocr = sample["ocr_bbox"]["coordinates"]  # [50, 4]
        sample["center_point_ocr"] = torch.stack(
            ((bbox_ocr[:, 0] + bbox_ocr[:, 2]) / 2 * sample_info["image_width"],
             (bbox_ocr[:, 1] + bbox_ocr[:, 3]) / 2 * sample_info["image_height"]), dim=1)

        return sample

    def add_all_bb_info(self, sample_info, sample):
        # 1. add bounding boxes info of faster-rcnn features
        # following this order: left, down, right, upper
        bbox_rcnn = sample["image_info_0"]["boxes"]  # [100, 4]
        sample["bb_rcnn"] = torch.from_numpy(bbox_rcnn)

        # 2. add all info of ResNet spatial features
        # generate coordinates of the grids in feature map
        grid_num = 14
        x_block = sample_info['image_width'] / grid_num
        y_block = sample_info['image_height'] / grid_num
        x1 = self.floatrange(0, sample_info["image_width"] - x_block, grid_num)
        y1 = self.floatrange(0, sample_info["image_height"] - y_block, grid_num)
        x2 = self.floatrange(x_block, sample_info["image_width"], grid_num)
        y2 = self.floatrange(y_block, sample_info["image_height"], grid_num)
        x_grid1, y_grid1 = np.meshgrid(x1, y1)
        x_grid1 = x_grid1.reshape([-1])
        y_grid1 = y_grid1.reshape([-1])
        x_grid2, y_grid2 = np.meshgrid(x2, y2)
        x_grid2 = x_grid2.reshape([-1])
        y_grid2 = y_grid2.reshape([-1])

        sample['bb_resnet'] = torch.stack(tuple(torch.tensor([x_grid1[i], y_grid1[i], x_grid2[i], y_grid2[i]])
                                                for i in range(grid_num ** 2)))  # [196, 4]

        # 3. all info of OCR bboxes
        bbox_ocr = sample["ocr_bbox"]["coordinates"]  # [50, 4]
        sample["bb_ocr"] = torch.stack(
            (bbox_ocr[:, 0] * sample_info["image_width"],
             bbox_ocr[:, 1] * sample_info["image_height"],
             bbox_ocr[:, 2] * sample_info["image_width"],
             bbox_ocr[:, 3] * sample_info["image_height"]), dim=1)

        return sample

    def re_order(self, sample_list):
        index_table = list(np.argsort(sample_list["image_info_0"]["cls_scores"]))
        index_table = torch.LongTensor([99 - index_table.index(i) for i in range(100)]).unsqueeze(1)
        sample_list["image_feature_0"].scatter_(0, index_table.repeat(1, 2048), sample_list["image_feature_0"])
        sample_list["center_point_rcnn"].scatter_(0, index_table.repeat(1, 2), sample_list["center_point_rcnn"])
        sample_list["bb_rcnn"].scatter_(0, index_table.repeat(1, 4), sample_list["bb_rcnn"])

        return sample_list

    def add_ocr_details(self, sample_info, sample):
        if self.use_ocr:
            # Preprocess OCR tokens
            ocr_tokens = [self.ocr_token_processor({"text": token})["text"] for token in sample_info["ocr_tokens"]]
            # Get embeddings for tokens
            context = self.context_processor({"tokens": ocr_tokens})
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"]:] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({"info": sample_info["ocr_info"]})["bbox"]

        # sample["value_tokens"], sample["value_embeddings"], sample["mask_v"] = value_sieve(
        #     sample["context_tokens"],
        #     sample["context"],
        #     sample["mask_s"])

        return sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == self.context_processor.PAD_TOKEN:
                answer = "unanswerable"

            predictions.append({"question_id": question_id.item(), "answer": answer})

        return predictions

    @staticmethod
    def floatrange(start, stop, steps):
        return [start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)]
