# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/7/17 2:48 下午
"""
import tensorflow as tf

import models.model_factory as model_factory


class BaseModel(tf.keras.Model, metaclass=model_factory.Meta):
    """
    base model for all models
    """
    type_str: str = None
