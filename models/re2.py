# -*- coding: utf-8 -*-

"""
@Author: xiezizhe 
@Date: 2020/10/14 2:54 下午
"""
import math

import tensorflow as tf
import tensorflow.keras.layers as layers

import common.logger
import models.base_model as base_model
from models.model_name import ModelName

logger = common.logger.get_logger(__file__)


class Encoder(layers.Layer):

    def __init__(self, hidden_size, kernel_size, rate, enc_layers=2):
        super(Encoder, self).__init__()
        self.enc_layers = enc_layers
        self.dropout = layers.Dropout(rate)
        self.conv = [layers.Conv1D(hidden_size, kernel_size, activation='relu', padding='SAME')
                     for _ in range(enc_layers)]

    def call(self, x, mask, training):
        for i in range(self.enc_layers):
            x = mask * x
            if i > 0:
                x = self.dropout(x, training=training)
            x = self.conv[i](x)
        x = self.dropout(x, training=training)
        return x


class Alignment(layers.Layer):

    def __init__(self, hidden_size, rate):
        super(Alignment, self).__init__()
        self.temperature = tf.Variable(math.sqrt(1.0 / hidden_size))
        self.dense = layers.Dense(hidden_size, activation='relu')
        self.dropout = layers.Dropout(rate)

    def call(self, a, b, mask_a, mask_b, training):
        at_a = self.dense(self.dropout(a, training=training))
        at_b = self.dense(self.dropout(b, training=training))
        attention = tf.matmul(at_a, at_b, transpose_b=True) * self.temperature
        attention_mask = tf.matmul(mask_a, mask_b, transpose_b=True)
        attention = attention_mask * attention + (1 - attention_mask) * tf.float32.min
        attention_a = tf.nn.softmax(attention, axis=1)
        attention_b = tf.nn.softmax(attention, axis=2)

        feature_b = tf.matmul(attention_a, a, transpose_a=True)
        feature_a = tf.matmul(attention_b, b)
        return feature_a, feature_b


class Fusion(layers.Layer):

    def __init__(self, hidden_size, rate):
        super(Fusion, self).__init__()
        self.dense1 = layers.Dense(hidden_size, activation='relu')
        self.dense2 = layers.Dense(hidden_size, activation='relu')
        self.dense3 = layers.Dense(hidden_size, activation='relu')
        self.dense = layers.Dense(hidden_size, activation='relu')
        self.dropout = layers.Dropout(rate)

    def call(self, x, align, training):
        x1 = self.dense1(tf.concat([x, align], axis=-1))
        x2 = self.dense2(tf.concat([x, x - align], axis=-1))
        x3 = self.dense3(tf.concat([x, x * align], axis=-1))
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        return x


class RE2(base_model.BaseModel):
    """
    for test
    """
    type_str = ModelName.Semantic_matching

    def __init__(self, num_vocab, embedding_dim, enc_layers, hidden_size, kernel_size, rate, num_blocks, num_classes,
                 pretrained_vocab=None):
        super().__init__()
        if pretrained_vocab:
            logger.info("loading pretrained embedding matrix")
            self.embedding = layers.Embedding(num_vocab,
                                              embedding_dim,
                                              embeddings_initializer=tf.keras.initializers.Constant(pretrained_vocab),
                                              trainable=False)
        else:
            logger.info("creating embedding matrix")
            self.embedding = layers.Embedding(num_vocab, embedding_dim)
        self.blocks = [{
            'encoder': Encoder(hidden_size, kernel_size, rate, enc_layers),
            'alignment': Alignment(hidden_size, rate),
            'fusion': Fusion(hidden_size, rate),
        } for _ in range(num_blocks)]
        self.dropout = layers.Dropout(rate)
        self.dense1 = layers.Dense(hidden_size, activation='relu')
        self.dense2 = layers.Dense(num_classes)

    def _connection(self, x, res, i):
        if i == 1:
            x = tf.concat([res, x], axis=-1)  # res is embedding
        elif i > 1:
            hidden_size = int(x.shape[-1])
            x = (res[:, :, -hidden_size:] + x) * math.sqrt(0.5)
            x = tf.concat([res[:, :, :-hidden_size], x], axis=-1)  # former half of res is embedding
        return x

    def pooling(self, x, mask):
        return tf.reduce_max(mask * x + (1. - mask) * tf.float32.min, axis=1)

    def _features(self, a, b):
        return tf.concat([a, b, a * b, a - b], axis=-1)

    def predict(self, a, b, training):
        x = self._features(a, b)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

    def forward(self, a, b, mask_a, mask_b, training):
        a = self.embedding(a)
        a = self.dropout(a, training=training)

        b = self.embedding(b)
        b = self.dropout(b, training=training)

        res_a, res_b = a, b
        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self._connection(a, res_a, i)
                b = self._connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a, training)
            b_enc = block['encoder'](b, mask_b, training)
            a = tf.concat([a, a_enc], axis=-1)
            b = tf.concat([b, b_enc], axis=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b, training)
            a = block['fusion'](a, align_a, training)
            b = block['fusion'](b, align_b, training)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.predict(a, b, training)

    @tf.function(
        input_signature=[(tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                          tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                          tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.bool))]
    )
    def call(self, inputs):
        return self.forward(*inputs)
