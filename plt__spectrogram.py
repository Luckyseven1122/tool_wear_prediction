# -*- coding: utf-8 -*-
# @Time    : 2018/12/7 10:16
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : plt__specturogram.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import spectrogram, spectral

# root = 'tool3'

for root in ['tool1','tool2','tool3','tool4','tool5']:

    plc = pd.read_csv(f'data/{root}/plc.csv')
    plc['time'] = plc['time'].apply(lambda x: x.split(':')).\
        apply(lambda x: int(x[0])*3600+int(x[1])*60+int(x[2])+int(x[3])/1000)

    plt.figure(root)
    x = [1, 11] if root=='tool4' or root=='tool5' else [15, 26]
    for j in range(x[0], x[1]):
        # print(j)
        # data = pd.read_csv(f'data/{root}/{j}.csv')
        # data.fillna(0, inplace=True)
        # data.clip(-35, 35, inplace=True)
        # plt.subplot(711)
        # plt.plot(data['vibration_1'])
        # plt.xlim([0, data.shape[0]])
        # plt.subplot(311)
        # f, t, Sxx = spectrogram(data['vibration_1'], 25600, scaling='spectrum')
        # print(f'axis1: {list(filter(lambda x: x>42,list(np.where(Sxx.max(axis=1)>1)[0])))}')
        # Sxx = Sxx.clip(-50, 50)
        # plt.pcolormesh(t, f, Sxx)
        # plt.subplot(713)
        # plt.plot(data['vibration_2'])
        # plt.xlim([0, data.shape[0]])
        # plt.subplot(312)
        # f, t, Sxx = spectrogram(data['vibration_2'], 25600, scaling='spectrum')
        # print(f'axis2: {list(filter(lambda x: x>42,list(np.where(Sxx.max(axis=1)>1)[0])))}')
        # Sxx = Sxx.clip(-50, 50)
        # plt.pcolormesh(t, f, Sxx)
        # plt.subplot(715)
        # plt.plot(data['vibration_3'])
        # plt.xlim([0, data.shape[0]])
        # plt.subplot(313)
        # f, t, Sxx = spectrogram(data['vibration_3'], 25600, scaling='spectrum')
        # print(f'axis3: {list(filter(lambda x: x>42,list(np.where(Sxx.max(axis=1)>1)[0])))}')
        # Sxx = Sxx.clip(-50, 50)
        # plt.pcolormesh(t, f, Sxx)
        # plt.subplot(717)
        plcx = plc[plc['csv_no'] == j].copy()
        plcx['time'] = plcx['time']-plcx['time'].values[0]
        plt.plot(plcx['time'], plcx['spindle_load'])
    plt.xlim([0, 60.5])
    plt.ylim([0, 40])
plt.show()
