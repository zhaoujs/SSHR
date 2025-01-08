#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Process_data.py 
@Author  ：zly
@Date    ：2023/12/15 13:54
'''
from asyncio import sleep

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.__version__)
import seaborn
print(seaborn.__version__)

def scale_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    """
    This function takes in a numpy array and a target length and returns a numpy array of the same length, where each element is linearly interpolated from the original array.

    Parameters
    ----------
    arr: np.ndarray
        The numpy array to be scaled.
    target_length: int
        The desired length of the output numpy array.

    Returns
    -------
    np.ndarray
        The scaled numpy array.

    """
    orig_indices = np.linspace(0, len(arr) - 1, len(arr))
    target_indices = np.linspace(0, len(arr) - 1, target_length)

    scaled_arr = np.interp(target_indices, orig_indices, arr)
    return scaled_arr
seed = 1
def arr_random_change_small(arr):
    global seed
    # 获取数组的长度
    arr = np.array(arr)
    length = len(arr)
    np.random.seed(seed)
    seed += 1
    # 确定要选择并减半的元素的数量
    num_elements_to_halve = length // 3

    # 随机选择一半的元素的索引
    selected_indices = np.random.choice(length, size=num_elements_to_halve, replace=False)

    # 将选定的元素减半
    arr[selected_indices] = arr[selected_indices] * 1
    return arr

# 读取文件中的变量
filename_list = ["abalone918.txt", "kr-vs-k-three_vs_eleven.txt", "new-thyroid1.txt",
                 "poker-8_vs_6.txt", "shuttle-c2-vs-c4.txt", "yeast4.txt",
                 "ecoli-0-1_vs_2-3-5.txt","glass2.txt"
                 ]
noise_level_list = [0.05, 0.1, 0.15, 0.2]
target_length = 32
heatmap_list_ori = []
heatmap_list_noi = []
ori_avg = []
noi_avg = []
for index, filename in enumerate(filename_list):
    for noise_level in noise_level_list:
        noi_list = []
        with open('noise_con/{}.{}-noi'.format(filename, noise_level), 'r') as file:
            for line in file:
                noi_list.append(float(line.strip()))
        noi_list = arr_random_change_small(noi_list)
        noi_list = sorted(noi_list, reverse=True)
        heatmap_list_noi.append(scale_array(noi_list, target_length))
        # noi_list = np.array(noi_list).reshape((1, -1))
        # sns.heatmap(noi_list, cmap='Reds_r', annot=False, fmt='.2f', linewidths=0, cbar=False)

        # 读取文件中的变量
        ori_list = []
        with open('noise_con/{}.{}-ori'.format(filename, noise_level), 'r') as file:
            for line in file:
                ori_list.append(float(line.strip()))
        ori_list = sorted(ori_list, reverse=True)
        heatmap_list_ori.append(scale_array(ori_list, target_length))
        # ori_list = np.array(ori_list).reshape((1, -1))
        # sns.heatmap(ori_list, cmap='viridis', annot=False, fmt='.2f', linewidths=0, cbar=False)

        # 显示热力图
        # plt.show()
        ori_avg.append("{:.2f}".format(np.mean(ori_list)))
        # noi_avg.append("{:.2f}".format(np.mean(noi_list)))
        print("{}\t{}\t{}\t{}".format(filename, noise_level, np.mean(ori_list), np.mean(noi_list)))
    # heatmap_list_noi = np.array(heatmap_list_noi).reshape((1, -1))
# exit()
#
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 调整参数以适应图形
sns.heatmap(heatmap_list_noi, cmap='Reds', annot=False, linewidths=0.000, ax=axs[0], cbar_ax=cbar_ax)
for i in range(1, len(heatmap_list_noi)):
    if i % 4 ==0:
        axs[0].axhline(y=i, color='black', linestyle='-', linewidth=2)
axs[0].set_title('Noise')
# 清除 x 和 y 轴上的刻度线和标签
axs[0].set_xticks([])
axs[0].set_yticks([])
# axs[0].set_xticklabels([])
# axs[0].set_yticklabels([])

# plt.savefig('Noise.png', bbox_inches='tight', pad_inches=0.05, dpi=400)
# plt.show()
# plt.close()
# heatmap_list_ori = np.array(heatmap_list_ori).reshape((1, -1))
sns.heatmap(heatmap_list_ori, cmap='Reds', annot=False, linewidths=0.000, ax=axs[1], cbar_ax=cbar_ax, )
for i in range(1, len(heatmap_list_ori)):
    if i % 4 ==0:
        axs[1].axhline(y=i, color='black', linestyle='-', linewidth=2)
axs[1].set_title('Original')
axs[1].set_xticks([])
# axs[1].set_yticks(noi_avg)

#
# 这边设置y坐标的标签
y_tick = []
filename_no = 1
for i in range(0, len(heatmap_list_ori)):
    if i % 4 == 1:
        y_tick.append("Data-{}   ".format(filename_no))
        filename_no += 1
    else:
        y_tick.append("   ")
axs[1].set_yticklabels(labels = y_tick, rotation=0,ha='right')
# 设置刻度的长度
axs[1].tick_params(length=0)

# plt.tight_layout(0.7)
plt.savefig('Original.png', bbox_inches='tight', pad_inches=0.05, dpi=800)
plt.show()
plt.close()
