# -*- coding: utf-8 -*-
# @Time    : 2018/12/9 18:28
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : cnnmodel.py
import tensorflow as tf
import numpy as np
from config import random_seed, gpu_device
from utils import parse_function
from tensorflow.contrib.rnn import GRUCell,GRUBlockCellV2


def conv2d(inputs, num_filters, filter_size, activation=None, padding='same', strides=1, l1_scale=0.0):
    out = tf.layers.conv2d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size, data_format='channels_first',
        strides=strides,
        activation=activation,
        kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=l1_scale))
    return out


def average_pool2d(inputs, pool_size, padding='valid', strides=2):
    out = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=pool_size, strides=strides, padding=padding, data_format='channels_first')
    return out


def dense(X, size, activation=None, l1_scale=0.0):
    out = tf.layers.dense(X, units=size, activation=activation,
                          kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=l1_scale))
    return out


def make_encoder(sequence, output_dim, seed=0):
    rnn_cell=GRUCell(num_units=output_dim,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,dtype=tf.float32, seed=seed),
                bias_initializer=tf.zeros_initializer())
    rnn_out, rnn_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=tf.transpose(sequence, [0, 1, 2]),
        initial_state=rnn_cell.zero_state(tf.shape(sequence)[0], dtype=tf.float32),
        time_major=False)
    return rnn_state


class CnnModel:
    def __init__(self, param):
        self.lr = 1e-3
        self.batch_size = 128
        self.epoch = 4
        self.l1_scale = 0.0001
        self.seed = random_seed

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.device(gpu_device):
            with graph.as_default():
                self.place_lr = tf.placeholder(tf.float32, shape=())
                self.filenames = tf.placeholder(tf.string, shape=())
                dataset = tf.data.Dataset.list_files(self.filenames, seed=256)
                dataset = tf.data.TFRecordDataset(dataset) \
                    .repeat() \
                    .apply(tf.contrib.data.map_and_batch(map_func=parse_function, batch_size=64, num_parallel_calls=6)) \
                    .prefetch(buffer_size=64)
                self.iterator = dataset.make_initializable_iterator()

                place_x, place_y = self.iterator.get_next()
                place_x = tf.cast(tf.reshape(place_x, [-1, 3*50, 96]), tf.float32)
                # place_x = place_x[:, :, 16:25]
                place_y = tf.reshape(place_y, [-1, 1])
                # out = conv2d(place_x, num_filters=64, filter_size=(3, 3), activation='relu', l1_scale=self.l1_scale)
                # out = average_pool2d(out, pool_size=(1, 2), strides=(1, 2))
                # out = conv2d(out, num_filters=128, filter_size=(3, 3), activation='relu', l1_scale=self.l1_scale)
                # out = tf.reduce_mean(out, axis=1)
                out = make_encoder(place_x, output_dim=64)
                out = tf.reshape(out, [-1, 10*out.shape[1]])
                out = dense(out, 64, activation='relu', l1_scale=self.l1_scale)
                self.out = dense(out, 1, l1_scale=self.l1_scale)

                # Eri = place_y-self.out*10
                # Eri = tf.where((Eri >= -1) & (Eri < 0), tf.zeros_like(Eri), Eri)
                # self.loss = 1-tf.reduce_mean(tf.where(Eri > 0, tf.exp(-0.0346573*Eri), tf.exp(0.138629*Eri)))
                self.loss = tf.losses.absolute_difference(place_y, 90+self.out)
                self.reg_loss = self.loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
                self.train_step = opt.minimize(self.reg_loss)
                init = tf.global_variables_initializer()
                self.saver = tf.train.Saver()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config, graph=graph)
            self.init = init

    def fit(self, train_path, valid_path):
        self.session.run(self.init)
        for j in range(40):
            self.session.run(self.iterator.initializer, feed_dict={self.filenames: train_path})
            for i in range(192):
                loss, reg_loss, _ = self.session.run([self.loss, self.reg_loss, self.train_step], feed_dict={self.place_lr: self.lr})
                if i % 64 == 0: print(f"Epoch: {i}, reg_loss: {reg_loss:.3f}, loss: {loss:.3f}, score: {1-loss:.3f}")
            for k, vp in enumerate(valid_path):
                loss = 0
                self.session.run(self.iterator.initializer, feed_dict={self.filenames: vp})
                for i in range(64): loss += self.session.run(self.loss)
                loss /= 64
                print(f"tool{k} loss: {loss:.3f}, score: {1-loss:.3f}")

    def predict(self, valid_path):
        result = []
        for vp in valid_path:
            out = 0
            self.session.run(self.iterator.initializer, feed_dict={self.filenames: vp})
            for i in range(64):
                pred = self.session.run(self.out)
                out += np.mean(pred)*10
            result.append(out/64)
        return result

    def save_model(self, path):
        self.saver.save(self.session, path)

    def load_model(self, path):
        self.saver.restore(self.session, path)
