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

# for root in ['tool1']:
for i,root in enumerate(['tool1','tool2','tool3']):

    plc = pd.read_csv(f'data/{root}/plc.csv')
    # plc['time'] = plc['time'].apply(lambda x: x.split(':')).\
    #     apply(lambda x: int(x[0])*3600+int(x[1])*60+int(x[2])+int(x[3])/1000)
    plc = plc[(plc['csv_no']>14) & (plc['csv_no']<=24) & (plc['spindle_load']>plc['spindle_load'].mean())]
    # plt.subplot(311+i)
    # plc['spindle_load'].plot.hist(1000, label=f'{root}')
    # plt.xlim([0,40])
    # plt.ylim([0,100])
    print(np.log(plc['spindle_load'].shape[0]), np.log(plc['spindle_load'].values).mean())

plc = pd.read_csv(f'data/tool5/plc.csv')
plc = plc[plc['spindle_load']>5]
print(np.log(plc['spindle_load'].shape[0]), np.log(plc['spindle_load'].values).mean())
# plc['spindle_load'].plot.hist(400, label=f'{root}')
# plt.xlim([0,40])
# plt.ylim([0,100])
# plt.legend()
# plt.show()

