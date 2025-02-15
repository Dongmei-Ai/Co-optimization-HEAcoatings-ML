from tools import *
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as ply

#shap重要性
def Shapimportance(target):
    # 加载模型
    models = joblib.load(f'ML_model_{target}.pkl')

    # 加载训练数据
    data = pd.read_csv(f'train_data_{target}.csv')
    data = data.iloc[:, 1:]

    # 计算SHAP值
    explainer = shap.TreeExplainer(models, data)
    shap_values = explainer.shap_values(data)
    return shap_values, data

shap_values_H, data_H = Shapimportance('H')
shap_values_M, data_M = Shapimportance('M')

fig, axes = plt.subplots(1,2)

plt.sca(axes[0])
shap.summary_plot(shap_values_H, data_H, show=False)  # 禁止自动显示图像
axes[0].set_title('Hardness')

plt.sca(axes[1])
shap.summary_plot(shap_values_M, data_M, show=False)  # 禁止自动显示图像
axes[1].set_title('Modulus')

plt.tight_layout()
plt.show()