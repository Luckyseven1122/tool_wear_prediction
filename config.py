# -*- coding: utf-8 -*-
# @Time    : 2018/12/8 10:18
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : config.py

processor = 6
t1_file = 'data/tool1/'
t2_file = 'data/tool2/'
t3_file = 'data/tool3/'
t4_file = 'data/tool4/'
t5_file = 'data/tool5/'

t1_nums = 47  # remove the last file of tool1
t2_nums = 48
t3_nums = 37
t4_nums = 10
t5_nums = 10
acc_freq = 25600  # Hz

rsv_fb = [[0, 44], [74, 80]]  # 50 interest freq band  0~43, 74~79
rsv_fl = 50
chunk_size = 96  # time axis length
clip_time = 35  # clip value in time axis
clip_freq = 50  # clip value in freq axis
chunk_reserve_th = 1  # chunks filter threshold in freq axis

toollife = [240, 240, 185]
first_file_time = 5  # the used time of first file of each tool is 5 min

train_size = 4096*20  # total train size
train_files = 20  # file nums of each tool
fsample = 4096  # samples in each file
random_seed = 256
gpu_device = "/gpu:1"
cnn_param = {
    'epoch': 4,
    'lr': 3e-3,
    'batch_size': 128,
    'input_dim': 183,
    'dense1': 1024,
    'dense2': 256,
    'dense3': 256,
    'l1_scale': 0.000001
}
