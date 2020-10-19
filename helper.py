# -*- coding: utf-8 -*-

"""
@Author: xiezizhe
@Date: 14/10/2020 上午12:36
"""

import math
import os
import shutil

import tensorflow as tf
import tqdm

import common.data_io as data_io
import common.logger
import common.utils as utils
import common.vocab
from models.re2 import RE2

logger = common.logger.get_logger(__name__)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr, min_lr, warmup_steps, decay_steps, decay_rate):
        super(CustomSchedule, self).__init__()

        self.warmup_steps = warmup_steps
        self.lr = lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, global_step):
        return tf.cond(tf.less(global_step, self.warmup_steps),
                       true_fn=lambda: self.min_lr + (self.lr - self.min_lr) / max(1, self.warmup_steps) * global_step,
                       false_fn=lambda: tf.maximum(self.min_lr, self.lr * self.decay_rate ** tf.floor(
                           (global_step - self.warmup_steps) / self.decay_steps)))


class Helper(object):

    def samples2steps(self, warmup_samples, batch_size):
        return int(math.ceil(warmup_samples / batch_size))

    def __init__(self, recreate=False):
        argparser, _ = utils.init_argsparser()
        config = utils.init_config(argparser.config)
        self.argparser = argparser
        self.config = config

        self.optimizer = tf.optimizers.Adam(learning_rate=CustomSchedule(
            lr=config.lr,
            min_lr=config.min_lr,
            warmup_steps=self.samples2steps(config.lr_warmup_samples, config.batch_size),
            decay_steps=self.samples2steps(config.lr_decay_samples, config.batch_size),
            decay_rate=config.decay_rate
        ))
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

        self.data, vocab = data_io.load_snli(path=argparser.snli_path, args=argparser, config=config)
        config.num_vocab = len(vocab)
        logger.info(f"config:{config}")

        # load embedding
        embedding = common.vocab.load_embeddings(args=argparser, config=config, vocab=vocab)

        self.model = RE2(num_vocab=config.num_vocab,
                         embedding_dim=config.embedding_dim,
                         enc_layers=config.enc_layers,
                         hidden_size=config.hidden_size,
                         kernel_size=config.kernel_size,
                         rate=config.rate,
                         num_blocks=config.num_blocks,
                         num_classes=config.num_classes,
                         pretrained_vocab=embedding)

        a, b, mask_a, mask_b, _ = list(self.data.train.take(1).as_numpy_iterator())[0]
        self.model((a, b, tf.expand_dims(mask_a, axis=-1), tf.expand_dims(mask_a, axis=-1), False))
        self.model.summary()

        if recreate:
            shutil.rmtree(os.path.join(os.path.dirname(__file__), self.argparser.output_dir), ignore_errors=True)

        checkpoint_path = os.path.join(os.path.dirname(__file__), self.argparser.output_dir)
        ckpt = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        self.initial_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logger.info('Latest checkpoint restored from {} !!'.format(self.ckpt_manager.latest_checkpoint))
            self.initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    def train(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        @tf.function
        def train_step(a, b, mask_a, mask_b, y):
            with tf.GradientTape() as tape:
                mask_a = tf.expand_dims(mask_a, axis=-1)
                mask_b = tf.expand_dims(mask_b, axis=-1)
                o = self.model((a, b, mask_a, mask_b, True))
                loss = self.scce(y, o)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            gradients, gnorm = tf.clip_by_global_norm(gradients, self.config.grad_clipping)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(y, o)

        best_so_far = tf.Variable(0, dtype=tf.float32)
        for epoch in range(self.initial_epoch, self.config.epochs):
            progress = tqdm.tqdm(total=self.data.train_size, desc='Batch', position=0)

            for (batch_num, inputs) in enumerate(self.data.train):
                progress.update(self.config.batch_size)
                train_step(*inputs)
                if batch_num > 0 and batch_num % 500 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch_num} '
                          f'Loss {self.train_loss.result()} Accu {self.train_accuracy.result()}')

            ckpt_save_path = self.ckpt_manager.save()
            logger.info(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path} '
                        f'Loss: {self.train_loss.result()} Accu: {self.train_accuracy.result()}')

            # evaluate model after each epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            for (batch_num, inputs) in enumerate(self.data.dev):
                mask_a = tf.expand_dims(inputs[2], axis=-1)
                mask_b = tf.expand_dims(inputs[3], axis=-1)
                o = self.model.call(inputs[0], inputs[1], mask_a, mask_b, training=False)
                self.train_accuracy(inputs[4], o)
            logger.info(f"dev accu results: {self.train_accuracy.result()}")

            if tf.greater(self.train_accuracy.result(), best_so_far):
                best_so_far = self.train_accuracy.result()
                output = os.path.join(os.path.dirname(__file__), self.argparser.output_dir, "final")
                shutil.rmtree(output, ignore_errors=True)
                if not os.path.exists(output):
                    os.makedirs(output)
                tf.saved_model.save(self.model, output)

        logger.info("train completed")

    def test(self):
        self.train_accuracy.reset_states()
        for (batch_num, inputs) in enumerate(self.data.test):
            mask_a = tf.expand_dims(inputs[2], axis=-1)
            mask_b = tf.expand_dims(inputs[3], axis=-1)
            o = self.model.call((inputs[0], inputs[1], mask_a, mask_b, False))
            self.train_accuracy(inputs[4], o)
        logger.info(f"test accu results: {self.train_accuracy.result()}")
