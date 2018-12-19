# -*- coding: utf-8 -*-
# @Time    : 2018/12/6 20:04
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : main.py
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

plc = pd.read_csv('data/tool1/plc.csv')

fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

for i in range(7, 9):
    tmp = plc[plc['csv_no'] == i]
    ax.plot(tmp['x'], tmp['y'], tmp['z'], label=f'no {i}')
plt.legend()
plt.show()
