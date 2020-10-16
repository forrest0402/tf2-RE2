# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/7/14 4:37 下午
"""
import logging
import logging.handlers
import os
import pathlib

# LOG_LEVER = logging.ERROR
# LOG_LEVER = logging.INFO
LOG_LEVER = logging.DEBUG

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, '..', 'log'))


def get_logger(name):
    """
    获取logger
    Args:
        name:

    Returns:

    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVER)

    formatter = logging.Formatter('%(asctime)s - %(name)s:[line:%(lineno)d] - %(levelname)s - %(message)s')

    pathlib.Path(MAIN_PATH).mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler("{}/semantic_matching.log".format(MAIN_PATH), mode="a",
                                                        maxBytes=104857600)

    # file_handler = TimedRotatingFileHandler("{}/bqa_server.log".format(MAIN_PATH),
    #                                     when="H", interval=1, backupCount=168)
    file_handler.setFormatter(formatter)

    console_handle = logging.StreamHandler()
    console_handle.setFormatter(formatter)

    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handle)

    return logger
