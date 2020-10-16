# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/7/17 2:49 下午
"""
from enum import Enum

import common.logger
from models.model_name import ModelName

models = dict()
logger = common.logger.get_logger(__name__)


def register_class(type_str: str, target_class):
    """
    automatically register model class to model factory
    Args:
        type_str:
        target_class:

    Returns:

    """
    if not type_str:
        return
    logger.info(f"register class[{target_class.__name__}][{type_str}]")
    models[type_str] = target_class


def get_model(type_str: Enum):
    """
    get model by model name
    Args:
        type_str:

    Returns:

    """
    if type_str not in ModelName:
        raise ValueError(f"cannot found {type_str} in ModelName")
    return models.get(type_str)


class Meta(type):
    """
    Meta class
    """
    _instances = {}

    def __new__(cls, name, bases, ds_dict):
        ds = super().__new__(cls, name, bases, ds_dict)
        register_class(ds_dict['type_str'], ds)
        return ds

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Meta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
