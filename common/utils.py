# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/7/14 4:49 下午
"""
import argparse
import configparser
import os

import bean

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, '..'))


def init_argsparser():
    """

    Returns:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=os.path.join(MAIN_PATH, "conf/app.conf"), help="配置文件路径")
    parser.add_argument("-v", "--version", default=False, help="查看文件版本", action="store_true")
    parser.add_argument("--console_enable", default=True, action="store_true")

    parser.add_argument('--output_dir', default='save_model/re2')
    parser.add_argument('--embedding_mode', default='freq')
    parser.add_argument('--lower_case', default=True)
    parser.add_argument('--snli_path', default='./data/snli')
    parser.add_argument('--pretrained_embeddings', default='./data/glove.840B.300d.txt')

    return parser.parse_known_args()


def init_config(path: str) -> bean.Config:
    """

    Args:
        path:

    Returns:

    """
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    args = {**{k: v for k, v in config['SNLI'].items()}, **{k: v for k, v in config['Training'].items()}}
    return bean.Config(**args).type_infer()
