# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from maskrcnn benchmark
from collections import defaultdict, deque

import torch


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def get_latest(self):
        return self.deque[-1]


class Meter:
    def __init__(self, delimiter=", "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, update_dict):
        for k, v in update_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dim() != 0:
                    v = v.mean()
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_from_meter(self, meter):
        for key, value in meter.meters.items():
            assert isinstance(value, SmoothedValue)
            self.meters[key] = value

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def get_scalar_dict(self):
        scalar_dict = {}
        for k, v in self.meters.items():
            scalar_dict[k] = v.get_latest()

        return scalar_dict

    def get_useful_dict(self):
        scalar_dict = {"loss": {}, "accuracy": {}}
        for name, meter in self.meters.items():
            if "loss" in name:
                if "val" in name:
                    # scalar_dict["loss"]["val"] = (sum(meter.series[:-1]) * 128 + meter.series[-1] * 8) / 5e3
                    scalar_dict["loss"]["val"] = meter.global_avg
                elif "train" in name:
                    scalar_dict["loss"]["train"] = meter.median
            elif "accuracy" in name:
                if "val" in name:
                    # scalar_dict["accuracy"]["val"] = (sum(meter.series[:-1]) * 128 + meter.series[-1] * 8) / 5e3
                    scalar_dict["accuracy"]["val"] = meter.global_avg
                elif "train" in name:
                    scalar_dict["accuracy"]["train"] = meter.median
        return scalar_dict

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if "train" in name:
                loss_str.append(
                    "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
                )
            else:
                loss_str.append("{}: {:.4f}".format(name, meter.global_avg))

        return self.delimiter.join(loss_str)
