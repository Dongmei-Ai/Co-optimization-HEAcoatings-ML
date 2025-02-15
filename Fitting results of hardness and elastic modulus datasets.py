import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d.proj3d import transform

from tools import *
import matplotlib.pyplot as plt

def PLT(target):
    gan_model = 'dhgcgan'
    path = f'data_{target}.csv'

    # 加载原始数据集
    property, craft, composition = data_load_org(path=path)

    # 原始数据集标准化（工艺和成分）
    craft_norm = Craft_norm(craft)
    composition_norm = comp_norm(composition)
    feature_Nodescriptor = pd.concat([craft_norm, composition_norm], axis=1)

    # 划分训练集和测试集
    X_train_org, X_test_org, y_train_org, y_test_org = split_data(feature=feature_Nodescriptor, property=property)

    # 在原始训练集上训练模型，测试集上测试模型，得到R2
    trainset_r2, testset_r2 = train_model(X_train=X_train_org, X_test=X_test_org, y_train=y_train_org,
                                          y_test=y_test_org)
    #print(trainset_r2, testset_r2)

    # 原始数据集计算描述符，描述符标准化
    descriptor = compute_descriptor(composition=composition_norm, target=target)
    descriptor_norm = Descriptor_norm(descriptor=descriptor, target=target)
    feature_descriptor = pd.concat([feature_Nodescriptor, descriptor_norm], axis=1)

    # 划分训练集和测试集
    X_train_org_des, X_test_org_des, y_train_org_des, y_test_org_des = split_data(feature=feature_descriptor,
                                                                                  property=property)

    # 加载生成数据集
    # 生成数据集标准化（主要是描述符！）
    # 原始训练集和生成数据集合并
    # 在增强后的数据集上训练模型，测试集上测试模型，得到R2
    '''
    best_gen_X, best_gen_y = choose_sample(target=target, gan_model='cgan',X_train_org_des=X_train_org_des, y_train_org_des=y_train_org_des,
                  X_test_org_des=X_test_org_des, y_test_org_des=y_test_org_des)
    best_gen_X, best_gen_y = choose_sample(target=target, gan_model='dhgcgan',X_train_org_des=X_train_org_des, y_train_org_des=y_train_org_des,
                  X_test_org_des=X_test_org_des, y_test_org_des=y_test_org_des)

    #超参数优化
    nsga(X_train=best_gen_X, X_test=X_test_org_des, y_train=best_gen_y, y_test=y_test_org_des)
    '''
    # 训练模型
    # 生成数据集标准化（主要是描述符！）
    # 生成数据集标准化（主要是描述符！）
    # 原始训练集和生成数据集合并
    print(f'{target}数据集R2:')
    # 测试模型在训练集和测试集上的R2
    train_R2, _= model_test(target)

    y_train_org_des, org_train_pred, property_gen, gen_train_pred, y_test_org_des, test_pred = model_train(
        target=target, gan_model=gan_model, X_train_org_des=X_train_org_des, y_train_org_des=y_train_org_des,
        X_test_org_des=X_test_org_des, y_test_org_des=y_test_org_des)
    #print(y_train_org_des, org_train_pred, property_gen, gen_train_pred, y_test_org_des, test_pred)

    return y_train_org_des, org_train_pred, property_gen, gen_train_pred, y_test_org_des, test_pred

H_org_train, H_org_train_pred, H_gen_y, H_gen_pred, H_org_test, H_org_test_pred = PLT(target='H')
M_org_train, M_org_train_pred, M_gen_y, M_gen_pred, M_org_test, M_org_test_pred = PLT(target='M')

indices_H = [24,4,13,8,6,11,3,20,16]
indices_M = [10,5,19,14,6,13,15,11,0]

test_plt_H = H_org_test.iloc[indices_H]
test_plt_H_pred = H_org_test_pred[np.array(indices_H)]
test_plt_M = M_org_test.iloc[indices_M]
test_plt_M_pred = M_org_test_pred[np.array(indices_M)]

labes_H = ['VNbCrWTa', 'NbWTaHf', 'MoWTa', 'ZrVTiCrW', 'AlNbTiCrWTa', 'VNbMoWTa', 'ZrNbTiCrMo', 'CuAlFeCoNiCr', 'AlFeCoNiCr']
labes_M = ['NbWTa', 'VTiCrWTa', 'AlVNbMoWTa', 'VNbMoWTa', 'AlZrNbCrMo', 'NbTiCrWTa', 'CuAlFeZrVCoNI', 'ZrNbTiMo', 'FeZrNbTiTa']

fig, axes = plt.subplots(2,2)

axes[0,0].scatter(H_gen_y, H_gen_pred, color='lightblue', marker='o', label='Augmented dataset')
axes[0,0].scatter(H_org_train, H_org_train_pred, color='darkblue', marker='s', label='Original training set')
axes[0,0].scatter(H_org_test, H_org_test_pred, color='red', marker='^', label='Original testset')
axes[0,0].set_xlabel('Experimental hardness(GPa)')
axes[0,0].set_ylabel('Predicted hardness(GPa)')
axes[0,0].legend()

axes[0,1].scatter(M_gen_y, M_gen_pred, color='lightblue', marker='o', label='Augmented dataset')
axes[0,1].scatter(M_org_train, M_org_train_pred, color='darkblue', marker='s', label='Original training set')
axes[0,1].scatter(M_org_test, M_org_test_pred, color='red', marker='^', label='Original testset')
axes[0,1].set_xlabel('Experimental modulus(GPa)')
axes[0,1].set_ylabel('Predicted modulus(GPa)')
axes[0,1].legend()

bar_width = 0.35
x_pos = np.arange(len(labes_H))
axes[1,0].bar(x_pos-bar_width/2, test_plt_H_pred, width=bar_width, label='Experimental values', color='lightblue')
axes[1,0].bar(x_pos+bar_width/2, test_plt_H, width=bar_width, label='Value predicted by model', color='darkblue')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(labes_H, rotation=30, ha='right')
axes[1,0].set_ylabel('Hardness(GPa)')
axes[1,0].legend()

x_pos = np.arange(len(labes_M))
axes[1,1].bar(x_pos-bar_width/2, test_plt_M_pred, width=bar_width, label='Experimental values', color='lightcoral')
axes[1,1].bar(x_pos+bar_width/2, test_plt_M, width=bar_width, label='Value predicted by model', color='darkred')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels(labes_M, rotation=30, ha='right')
axes[1,1].set_ylabel('Modulus(GPa)')
axes[1,1].legend()

labels = ['(a)', '(b)', '(c)', '(d)']

for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.1, labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()