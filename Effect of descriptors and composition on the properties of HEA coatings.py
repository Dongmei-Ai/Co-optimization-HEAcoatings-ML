import pandas as pd
import  matplotlib.pyplot as plt

data = pd.read_csv('found_system5.csv')

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # 1 行 2 列

# 第一张图：按 W 是否为 0 分类
axes[0].scatter(data[data["W"] != 0]["var:evaporation_heat"], data[data["W"] != 0]["hardness"], color="darkblue", label="W ≠ 0", alpha=0.7)
axes[0].scatter(data[data["W"] == 0]["var:evaporation_heat"], data[data["W"] == 0]["hardness"], color="lightblue", label="W = 0", alpha=0.7)
axes[0].set_xlabel("var:evaporation_heat")
axes[0].set_ylabel("Hardness")
axes[0].legend()

# 第二张图：按 Zr 是否为 0 分类
axes[1].scatter(data[data["Zr"] != 0]["var:gs_energy"], data[data["Zr"] != 0]["modulus"], color="darkred", label="Zr ≠ 0", alpha=0.7)
axes[1].scatter(data[data["Zr"] == 0]["var:gs_energy"], data[data["Zr"] == 0]["modulus"], color="lightcoral", label="Zr = 0", alpha=0.7)
axes[1].set_xlabel("var:gs_energy")
axes[1].set_ylabel("Modulus")
axes[1].legend()

# 调整布局并显示
plt.tight_layout()
plt.show()

