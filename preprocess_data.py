# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 10:22
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : preprocess_data.py
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.signal import spectrogram


def preprocess_spectrogram(acc_freq, clip_freq, clip_time, data_info):
    tool_file, tool_nums = data_info
    sample = []
    for i in range(1, tool_nums+1):
        print(i)
        data = pd.read_csv(tool_file+f'{i}.csv')
        data.fillna(0, inplace=True)
        data.clip(-clip_time, clip_time, inplace=True)  # time clip
        # convert data to spectrogram
        f, t, Sxx1 = spectrogram(data['vibration_1'], acc_freq, scaling='spectrum')
        f, t, Sxx2 = spectrogram(data['vibration_2'], acc_freq, scaling='spectrum')
        f, t, Sxx3 = spectrogram(data['vibration_3'], acc_freq, scaling='spectrum')
        Sxx = np.array((Sxx1[1:], Sxx2[1:], Sxx3[1:]), np.float32)  # [3, 128, len(t)]
        Sxx = np.clip(Sxx, -clip_freq, clip_freq)  # freq clip
        sample.append(Sxx)
    return sample


def get_available_index(chunk_size, chunk_reserve_th, data):
    sum_data = data.sum(axis=(0, 1))
    valid_index = []
    for i in range(0, sum_data.shape[0]-chunk_size):
        if sum_data[i:i+chunk_size].max() > chunk_reserve_th:
            valid_index.append(i)
    return valid_index


def get_train_index(available_index, train_size):
    L = len(available_index)
    index_len = np.array(list(map(len, available_index)))
    # convert available_index to numpy array
    available_index_np = np.empty([L, index_len.max()])
    for i in range(L):
        available_index_np[i, :index_len[i]] = available_index[i]
    # get train index
    train_index = np.random.rand(train_size, 11)  # 1 file point + 10 index
    train_index[:, 0] = np.floor(train_index[:, 0]*(L-9))
    for i in range(1, 11):
        train_index[:, i] = np.floor(train_index[:, i]*index_len[(train_index[:, 0]+i-1).astype(np.int32).tolist()])
        train_index[:, i] = available_index_np[(train_index[:, 0]+i-1).astype(np.int32).tolist(),
                                               train_index[:, i].astype(np.int32).tolist()]
    return train_index.astype(np.int32)
