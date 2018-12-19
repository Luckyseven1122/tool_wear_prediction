# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 10:16
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : main.py
import pickle
import numpy as np
import tensorflow as tf
from functools import partial
from multiprocessing import Pool
from config import t1_file, t1_nums, t2_file, t2_nums, t3_file, t3_nums, chunk_size, acc_freq, \
    chunk_reserve_th, clip_freq, clip_time, processor, train_size, toollife, cnn_param, \
    rsv_fb, rsv_fl, train_files, fsample, t4_file, t4_nums, t5_nums, t5_file
from preprocess_data import preprocess_spectrogram, get_available_index, get_train_index
from utils import timer, np_to_tfrecords
from cnnmodel import CnnModel


def main():
    # with Pool(processes=processor) as pool, timer('preprocess'):
    #     data_info = [[t1_file, t1_nums], [t2_file, t2_nums], [t3_file, t3_nums]]
    #     # data_info = [[t4_file, t4_nums], [t5_file, t5_nums]]
    #     data = pool.map(partial(preprocess_spectrogram, acc_freq, clip_freq, clip_time), data_info)
    #     pickle.dump(data, open('data/preprocess_spectrogram', 'wb'))
    #
    # with Pool(processes=processor) as pool, timer('generate data'):
    #     datax = pickle.load(open('data/preprocess_spectrogram', 'rb'))
    #     flags = ['valid']
    #     file_nums = [1]
    #     for flag, fnum in zip(flags, file_nums):
    #         for L in range(2,3):
    #             data = datax[L]
    #             train_index = pool.map(partial(get_available_index, chunk_size, chunk_reserve_th), data)
    #             train_index = get_train_index(train_index, train_size)
    #
    #             X_train = np.empty([fsample, 10, 3, rsv_fl, chunk_size], np.float16)  # [4096,10,3,50,96]
    #             for i in range(fnum):  # 20
    #                 for j in range(fsample):  # 4096
    #                     index = train_index[i*fsample+j, :]
    #                     for k in range(10):
    #                         X_train[j, k] = np.concatenate((
    #                             data[index[0] + k][:, rsv_fb[0][0]:rsv_fb[0][1], index[k + 1]: index[k + 1] + chunk_size],
    #                             data[index[0] + k][:, rsv_fb[1][0]:rsv_fb[1][1], index[k + 1]: index[k + 1] + chunk_size]
    #                         ), axis=1)
    #                 y_train = (toollife[L]-(train_index[i*fsample:(i+1)*fsample, 0]+10)*5).astype(np.float32)
    #                 np_to_tfrecords(X_train.reshape(fsample, -1), y_train.reshape(fsample, 1),
    #                                 f'data/tool{L+1}_{flag}_data{i}')

    # init model
    model = CnnModel(cnn_param)

    with timer('fit'):
        model.fit(train_path='data/tool1_train_data*.tfrecords',
                  valid_path=['data/tool1_valid_data*.tfrecords',
                              'data/tool2_valid_data*.tfrecords',
                              'data/tool3_valid_data*.tfrecords'])
        model.save_model('model/cnnmodel.ckpt')

    # with timer('predict'):
    #     model.load_model('model/cnnmodel.ckpt')
    #     result = model.predict(valid_path=['data/tool5_valid_data*.tfrecords',
    #                                        'data/tool4_valid_data*.tfrecords'])
    #     print(result)


if __name__ == '__main__':
    main()
