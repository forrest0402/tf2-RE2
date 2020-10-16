# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/10/10 6:26 下午
"""
import os
from typing import Set

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

import common.logger
from bean import Config, Dataset, Item
from common.vocab import Vocab

logger = common.logger.get_logger(__file__)


def snli_label_int(label: str) -> int:
    """

    Args:
        label:

    Returns:

    """
    if label == 'neutral':
        return 1
    if label == 'entailment':
        return 2
    if label == 'contradiction':
        return 0

    raise ValueError(f'invalid label string {label}')


def load_vocab(args, words: Set) -> Vocab:
    """

    Args:
        args:
        words:

    Returns:

    """
    output_dir = os.path.realpath(os.path.dirname(__file__))
    vocab_file = os.path.join(output_dir, 'vocab.txt')
    if not os.path.exists(vocab_file):
        vocab = Vocab.build(words,
                            lower=True, min_df=5, log=logger.info,
                            pretrained_embeddings=args.pretrained_embeddings,
                            dump_filtered=os.path.join(output_dir, 'filtered_words.txt'))
        vocab.save(vocab_file)
    else:
        vocab = Vocab.load(vocab_file)
    return vocab


def load_snli(args, config: Config, path: str = None) -> (Dataset, Vocab):
    """

    Args:
        args:
        config:
        path:

    Returns:

    """
    if not path:
        data = tfds.load('snli', shuffle_files=True)
        raise ValueError('Path is none')
    else:
        data, words = Dataset(), set()

        def read(suffix):
            with open(os.path.join(path, f'{suffix}.txt'), 'r', encoding='utf-8') as fr:
                d = []
                for line in fr.readlines():
                    try:
                        lines = line.strip('\n').split('\t')
                        label_int = int(lines[2])  # snli_label_int(lines[2])
                        text1 = lines[0].lower()
                        text2 = lines[1].lower()
                        words.update(text1.split())
                        words.update(text2.split())
                        d.append(Item(premise=text1, hypothesis=text2, label=label_int))
                    except ValueError:
                        pass
                    except Exception as e:
                        logger.exception(e)

                setattr(data, f'raw_{suffix}', d)
                setattr(data, f'{suffix}_size', len(d))
                logger.info(f'{suffix} size: {len(d)}')

        read('train')
        read('dev')
        read('test')
        # after reading, load vocabulary
        vocab = load_vocab(args=args, words=words)

        def preprocess(suffix):
            """
            tokenization, padding, create mask
            Args:
                suffix:

            Returns:

            """
            d = getattr(data, f'raw_{suffix}')

            def vectorize(i: Item):
                p = [vocab.index(w) for w in i.premise.split()[:config.max_len]]
                h = [vocab.index(w) for w in i.hypothesis.split()[:config.max_len]]
                mask_p = tf.expand_dims(tf.sequence_mask(len(p), dtype=tf.float32), axis=-1)
                mask_h = tf.expand_dims(tf.sequence_mask(len(h), dtype=tf.float32), axis=-1)
                p = tf.squeeze(pad_sequences([p], maxlen=config.max_len, padding="post", truncating="post"))
                h = tf.squeeze(pad_sequences([h], maxlen=config.max_len, padding="post", truncating="post"))
                mask_p = tf.squeeze(pad_sequences([mask_p], maxlen=config.max_len, padding="post", truncating="post"))
                mask_h = tf.squeeze(pad_sequences([mask_h], maxlen=config.max_len, padding="post", truncating="post"))
                return p, h, mask_p, mask_h, i.label

            def gen_series():
                for i in d:
                    yield vectorize(i)

            output_types = (tf.int32, tf.int32, tf.float32, tf.float32, tf.int32)
            dataset = tf.data.Dataset.from_generator(gen_series, output_types=output_types) \
                .shuffle(buffer_size=config.buffer_size) \
                .batch(config.batch_size, drop_remainder=True)
            setattr(data, f'{suffix}', dataset)

        preprocess('train')
        preprocess('dev')
        preprocess('test')
        print(list(data.train.take(1).as_numpy_iterator()))

    logger.info("load snli dataset completely")
    return data, vocab
