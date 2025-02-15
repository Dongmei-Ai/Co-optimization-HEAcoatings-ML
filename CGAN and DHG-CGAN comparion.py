import pandas as pd
from IPython.core.pylabtools import figsize
from deap.benchmarks.tools import rotate
from sympy import rotations

from tools import *
import numpy as np
import matplotlib.pyplot as plt

def r2_list(target):

    path = f'data_{target}.csv'
    #加载原始数据集
    property, craft, composition = data_load_org(path=path)

    #原始数据集标准化（工艺和成分）
    craft_norm = Craft_norm(craft)
    composition_norm = comp_norm(composition)
    feature_Nodescriptor = pd.concat([craft_norm, composition_norm], axis=1)

    #划分训练集和测试集
    X_train_org, X_test_org, y_train_org, y_test_org = split_data(feature=feature_Nodescriptor, property=property)

    #在原始训练集上训练模型，测试集上测试模型，得到R2
    trainset_r2, _ = train_model(X_train=X_train_org, X_test=X_test_org, y_train=y_train_org, y_test=y_test_org)
    r2_none = pd.Series(trainset_r2)
    #print(r2_none)

    #原始数据集计算描述符，描述符标准化
    descriptor = compute_descriptor(composition=composition_norm, target=target)
    descriptor_norm = Descriptor_norm(descriptor=descriptor, target=target)
    feature_descriptor = pd.concat([feature_Nodescriptor, descriptor_norm], axis=1)

    #划分训练集和测试集
    X_train_org_des, X_test_org_des, y_train_org_des, y_test_org_des = split_data(feature=feature_descriptor, property=property)

    #加载生成数据集
    # 生成数据集标准化（主要是描述符！）
    # 原始训练集和生成数据集合并
    # 在增强后的数据集上训练模型，测试集上测试模型，得到R2

    r2_cgan = choose_sample(target=target, gan_model='cgan', X_train_org_des=X_train_org_des,
                            y_train_org_des=y_train_org_des,
                            X_test_org_des=X_test_org_des, y_test_org_des=y_test_org_des)
    r2_cgan_average = r2_cgan.mean()
    # print(r2_cgan_average)

    r2_dhgcgan = choose_sample(target=target, gan_model='dhgcgan', X_train_org_des=X_train_org_des,
                               y_train_org_des=y_train_org_des,
                               X_test_org_des=X_test_org_des, y_test_org_des=y_test_org_des)
    r2_dhgcgan_average = r2_dhgcgan.mean()
    # print(r2_dhgcgan_average)
    return r2_none, r2_cgan_average, r2_dhgcgan_average

r2_none_H, r2_cgan_average_H, r2_dhgcgan_average_H = r2_list(target='H')
r2_none_M, r2_cgan_average_M, r2_dhgcgan_average_M = r2_list(target='M')


labels = ["AdaBoostRegressor", "RandomForestRegressor", "GradientBoostingRegressor", "BaggingRegressor", "ExtraTreesRegressor", "XGBRegressor"]

num_group = len(labels)
bar_width = 0.25
x = np.arange(num_group)

fig, axes = plt.subplots(1,2)

axes[0].bar(x-bar_width,r2_none_H, width=bar_width, label='Original dataset(H)')
axes[0].bar(x,r2_cgan_average_H, width=bar_width, label='CGAN augmented dataset(H)')
axes[0].bar(x+bar_width,r2_dhgcgan_average_H, width=bar_width, label='DHGCGAN augmented dataset(H)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=30)
axes[0].set_ylabel('R2')
axes[0].legend()

axes[1].bar(x-bar_width,r2_none_M, width=bar_width, label='Original dataset(M)')
axes[1].bar(x,r2_cgan_average_M, width=bar_width, label='CGAN augmented dataset(M)')
axes[1].bar(x+bar_width,r2_dhgcgan_average_M, width=bar_width, label='DHGCGAN augmented dataset(M)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=30)
axes[1].set_ylabel('R2')
axes[1].legend()

plt.tight_layout()
plt.show()