from tools import *
import pandas as pd
from itertools import combinations
from NSGA2 import *
import matplotlib.pyplot as plt
import numpy as np

'''
five_elements_systems = found_systems(5)
six_elements_systems = found_systems(6)
seven_elements_systems = found_systems(7)

result5 = process_file(file_path='found_system5.csv')
result6 = process_file(file_path='found_system6.csv')
result7 = process_file(file_path='found_system7.csv')

j = 1
for i in result5:
    nsga2(system=i, system_num=5, system_i=j)
    j +=1

j = 1
for i in result6:
    print(i)
    nsga2(system=i, system_num=6, system_i=j)
    j +=1

j = 1
for i in result7:
    print(i)
    nsga2(system=i, system_num=7, system_i=j)
    j +=1

'''
def System_path(i):
    systmes = []
    for j in range(1, 4):
        path = f'found_pareto/Final_generation_results{i}_{j}.csv'
        systmes.append(path)
    return systmes

system5 = System_path(5)
system6 = System_path(6)
system7 = System_path(7)


colors_5 = plt.cm.rainbow(np.linspace(0,0.3,len(system5)))
colors_6 = plt.cm.rainbow(np.linspace(0.3,0.6,len(system6)))
colors_7 = plt.cm.rainbow(np.linspace(0.6,0.9,len(system5)))

plt.figure(figsize=(8,6))
def PLT(systems_csv, colors, i):
    for file, color in zip(systems_csv, colors):
        df = pd.read_csv(file)
        label_str = ''.join(df.columns[3:3 + i])
        plt.scatter(df['Hardness'], df['Modulus'], color=color, label=label_str)
PLT(system5, colors_5, 5)
PLT(system6, colors_6, 6)
PLT(system7, colors_7, 7)

data = pd.read_csv('data.csv')
plt.scatter(data['Hardness'], data['Modulus'], color='gray', label='Original dataset')

plt.xlabel('Hardness(GPa)')
plt.ylabel('Modulus(GPa)')
plt.legend()
plt.show()