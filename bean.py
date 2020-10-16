# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/10/10 6:28 下午
"""
from dataclasses import dataclass
from typing import List

import tensorflow as tf


@dataclass
class Item(object):
    premise: str
    hypothesis: str
    label: int
    predicted: int = None

    def label_str(self) -> str:
        if self.label == 1:
            return 'neutral'
        if self.label == 2:
            return 'contradiction'
        if self.label == 0:
            return 'entailment'

        return 'unknown'


@dataclass
class Config(object):
    embedding_dim: int
    enc_layers: int
    kernel_size: int
    hidden_size: int
    rate: float
    num_blocks: int
    num_classes: int
    max_len: int

    epochs: int
    batch_size: int
    buffer_size: int
    grad_clipping: int
    lr: float
    min_lr: float
    decay_rate: float
    lr_decay_samples: int
    lr_warmup_samples: int

    num_vocab: int = None

    def _read(self, value):
        if not value:
            return value
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        return value

    def type_infer(self):
        for k, v in self.__dict__.items():
            self.__setattr__(k, self._read(v))
        return self


@dataclass
class Dataset(object):
    raw_train: List
    raw_dev: List
    raw_test: List

    train: tf.data.Dataset
    dev: tf.data.Dataset
    test: tf.data.Dataset

    train_size: int
    dev_size: int
    test_size: int

    def __init__(self):
        pass
