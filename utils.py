# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 10:24
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : utils.py
import time
import tensorflow as tf
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    # iterate over each sample, and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        d_feature = {
            'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X[idx].tobytes()])),
            'Y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y[idx].tobytes()])),
        }
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    if verbose:
        print("Serialized {:d} examples into {}".format(X.shape[0], result_tf_file))


def parse_function(example_proto):
    features = {"X": tf.FixedLenFeature((), tf.string),
                "Y": tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['X'], tf.float16)
    label = tf.decode_raw(parsed_features['Y'], tf.float32)
    return data, label


def load_tfrecords(srcfile):
    # tfrecords_to_np, which is used to validate np_to_tfrecords
    sess = tf.Session()
    dataset = tf.data.TFRecordDataset(srcfile)  # load tfrecord file
    dataset = dataset.map(parse_function)  # parse data into tensor
    dataset = dataset.repeat(2)  # repeat for 2 epoches
    dataset = dataset.batch(5)  # set batch_size = 5
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    while True:
        data, label = sess.run(next_data)
        print(data, label)
