# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 14/10/2020 上午12:46
"""

import os

import tensorflow as tf

import common.logger
import common.vocab
import helper

logger = common.logger.get_logger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f'{len(gpus)} Physical GPUs; {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        logger.exception(e)


def main():
    h = helper.Helper()
    h.test()


if __name__ == '__main__':
    main()
    logger.info('hello, world')
