import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from xenonpy.descriptor import Compositions
from scipy.stats import spearmanr
from xenonpy.datatools import preset
from deap import base, creator, tools, algorithms
from itertools import combinations
import matplotlib.pyplot as plt
import shap

models = {
        "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "BaggingRegressor": BaggingRegressor(random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(random_state=42)
}

#导入原始数据集
def data_load_org(path):

    #导入原始数据
    data = pd.read_csv(path)
    property = data.iloc[:,0]
    craft = data.iloc[:,1:4]
    composition = data.iloc[:,4:]
    return property, craft, composition

#导入生成数据集
def data_load_gen(path):
    #导入数据
    data = pd.read_csv(path)
    property = data.iloc[:,0]
    craft = data.iloc[:,1:4]
    composition = data.iloc[:,4:19]
    descriptor = data.iloc[:,19:]

    return property, craft, composition, descriptor

# 模型训练  输入：X_train, X_test, y_train, y_test  输出：trainset_r2, testset_r2
def train_model(X_train, X_test, y_train, y_test):
    trainset_r2 = []
    testset_r2 = []
    for name, model in models.items():
        # 十倍交叉验证测试模型
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        trainset_r2.append(scores.mean())
        # 训练集和测试集预测
        model.fit(X_train, y_train)
        #train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        #train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        testset_r2.append(test_r2)
    return trainset_r2, testset_r2

# 成分归一化函数
def comp_norm(composition):
    normalized_composition = composition.div(composition.sum(axis=1), axis=0)
    return normalized_composition

# 工艺归一化函数
def Craft_norm(craft):
    pressure = craft.iloc[:,0]
    bias = craft.iloc[:,1]
    flow = craft.iloc[:,2]

    pressure_norm = (pressure-0.08)/(2-0.08)
    bias_norm = (bias+200)/200
    flow_norm = (flow-6)/(100-6)

    craft_norm = pd.concat([pressure_norm, bias_norm, flow_norm], axis=1)

    return craft_norm

# 描述符归一化函数
def Descriptor_norm(descriptor, target):
    # 计算所有数据集的描述符并换取描述符最大最小值
    if target == 'H':
        num_f_valence = descriptor[['ave:num_f_valence']]
        num_f_valence_norm = num_f_valence/11.48
        evaporation_heat = descriptor[['var:evaporation_heat']]
        evaporation_heat_norm = (evaporation_heat-416)/(48024-416)
        num_unfilled = descriptor[['var:num_unfilled']]
        num_unfilled_norm = (num_unfilled-0.1659)/(5.6625-0.1659)
        vdw_radius_alvarez = descriptor[['var:vdw_radius_alvarez']]
        vdw_radius_alvarez_norm = (vdw_radius_alvarez-2.6946)/(162.5175-2.6946)

        descriptor_norm = pd.concat([num_f_valence_norm, evaporation_heat_norm, num_unfilled_norm, vdw_radius_alvarez_norm], axis=1)

    if target == 'M':
        thermal_conductivity = descriptor[['ave:thermal_conductivity']]
        thermal_conductivity_norm = (thermal_conductivity-31.2)/(155.57-31.2)
        polarizability = descriptor[['var:Polarizability']]
        polarizability_norm = (polarizability-0.71)/(17.867-0.706)
        gs_energy = descriptor[['var:gs_energy']]
        gs_energy_norm = (gs_energy-0.549)/(11.24-0.549)

        descriptor_norm = pd.concat([thermal_conductivity_norm, polarizability_norm, gs_energy_norm], axis=1)

    return descriptor_norm

#数据集划分
def split_data(feature, property, seed=42):

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature, property,test_size=0.2, random_state=seed)
    X_train = X_train.reset_index(drop=True)  #重置索引
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

#计算描述符
def compute_descriptor(composition, target):

    num = len(composition)

    preset.elements
    preset.elements_completed

    composition_dict = [] #用于存储字典形式的成分数据
    for i in range(0, num, 1):
        comps = composition.loc[i]
        a = dict(comps)
        composition_dict.append(a)

    # 使用Compositions类进行计算
    cal = Compositions()
    descriptor = cal.transform(composition_dict)

    # 将计算结果转换为Pandas DataFrame
    df_descriptor = pd.DataFrame(descriptor)

    #print(df_descriptor)

    if target == 'H':
        df_descriptor = df_descriptor[['ave:num_f_valence', 'var:evaporation_heat', 'var:num_unfilled', 'var:vdw_radius_alvarez']]

    if target == 'M':
        df_descriptor = df_descriptor[['ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy']]

    return df_descriptor

#超参数优化
def nsga(X_train, X_test, y_train, y_test):
    def evaluate(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf = params[0]
        if abs(n_estimators) <= 1:
            n_estimators = 20
        if n_estimators >= 200:
            n_estimators = 200
        if abs(max_depth) == 0:
            max_depth = 3
        if abs(max_depth) >= 20:
            max_depth = 20
        if abs(int(min_samples_split)) <= 1:
            min_samples_split = 2
        if abs(min_samples_split) >= 20:
            min_samples_split = 20
        if abs(int(min_samples_leaf)) == 0:
            min_samples_leaf = 1
        if abs(min_samples_leaf) >= 10:
            min_samples_leaf = 10

        # 使用给定的超参数配置构建模型
        model = ExtraTreesRegressor(n_estimators=abs(int(n_estimators)), max_depth=abs(int(max_depth)),
                                    min_samples_split=abs(int(min_samples_split)),
                                    min_samples_leaf=abs(int(min_samples_leaf)),
                                    random_state=42)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        train_r2 = scores.mean()
        # 训练模型
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        # 返回训练集和测试集上的R^2得分作为多目标优化的目标函数
        return train_r2, test_r2

    def attr_int_with_ranges():
        return np.random.randint(20, 200), np.random.randint(10, 25), np.random.randint(2, 20), np.random.randint(1, 10)

    #创建DEAP遗传算法所需的适应度函数和个体
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # 最大化训练集和测试集上的R^2得分
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", attr_int_with_ranges)# 使用新的函数来生成四个整数值
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册评价函数
    toolbox.register("evaluate", evaluate)

    # 定义遗传算法操作
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉操作
    toolbox.register("mutate", tools.mutUniformInt, low=[20, 10, 2, 1],
                 up=[200, 25, 20, 10], indpb=0.2)  # 变异操作，使用参数的最大范围
    toolbox.register("select", tools.selNSGA2)  # 选择操作

    # 创建初始种群
    population = toolbox.population(n=100)

    # 运行遗传算法进行优化
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.3, ngen=50, verbose=True)

    # 输出最优个体
    best_individuals = tools.selBest(population, k=100)
    for ind in best_individuals:
        print("Best Individual:", ind)
        print("Best Individual Fitness (Train R^2, Test R^2):", ind.fitness.values)

#生成数据集选择:输入：H/M，cgan/dhgcgan，原始训练集（X），原始训练集（y），原始测试集（X），原始测试集（y）。输出：生成模型在不同训练数据量下的训练集和测试集R2
def choose_sample(target, gan_model, X_train_org_des, y_train_org_des, X_test_org_des, y_test_org_des):
    r2 = []
    # 加载生成数据集
    for sample in range(50, 301, 50):
        property_gen, craft_gen, composition_gen, descriptor_gen = data_load_gen(
            path=f'Augemented_dataset_{target}/{target}_{gan_model}_generated_data_{sample}.csv')
        # 生成数据集标准化（主要是描述符！）
        descriptor_gen_norm = Descriptor_norm(descriptor_gen, target=target)
        feature_gen = pd.concat([craft_gen, composition_gen, descriptor_gen_norm], axis=1)

        # 原始训练集和生成数据集合并
        X_train_add = pd.concat([X_train_org_des, feature_gen], axis=0)
        y_train_add = pd.concat([y_train_org_des, property_gen], axis=0)

        # 在增强后的数据集上训练模型，测试集上测试模型，得到R2
        trainset_r2_add, testset_r2_add = train_model(X_train=X_train_add, X_test=X_test_org_des, y_train=y_train_add,
                                                      y_test=y_test_org_des)

        r2.append(trainset_r2_add)
        #print(f'生成样本数量等于{sample}时：')
        #print(trainset_r2_add, testset_r2_add)

        #if sample == 300:
        #    best_gen_X = X_train_add
        #    best_gen_y = y_train_add
    #return best_gen_X, best_gen_y
    r2_df = pd.DataFrame(r2)
    return r2_df

#训练模型
def model_train(target, gan_model, X_train_org_des, y_train_org_des, X_test_org_des, y_test_org_des):
    if target == 'M':
        model = ExtraTreesRegressor(n_estimators=2, max_depth=11,
                                  min_samples_split=8,
                                  min_samples_leaf=1,
                                  random_state=42)
    if target == 'H':
        model = ExtraTreesRegressor(n_estimators=2, max_depth=14,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      random_state=42)
    property_gen, craft_gen, composition_gen, descriptor_gen = data_load_gen(path=f'Augemented_dataset_{target}/{target}_{gan_model}_generated_data_300.csv')
    # 生成数据集标准化（主要是描述符！）
    descriptor_gen_norm = Descriptor_norm(descriptor_gen, target=target)
    feature_gen = pd.concat([craft_gen, composition_gen, descriptor_gen_norm], axis=1)

    # 原始训练集和生成数据集合并
    X_train_add = pd.concat([X_train_org_des, feature_gen], axis=0)
    y_train_add = pd.concat([y_train_org_des, property_gen], axis=0)

    model.fit(X_train_add, y_train_add)

    train_preds = model.predict(X_train_add)
    train_r2 = r2_score(y_train_add, train_preds)

    test_preds = model.predict(X_test_org_des)


    # 保存模型
    #joblib.dump(model, filename=f'ML_model_{target}.pkl')
    #print('模型已保存')

    # 构建csv文件，用于绘制预测散点图  原始训练集、生成训练集、原始测试集
    org_train_pred = model.predict(X_train_org_des)
    gen_train_pred = model.predict(feature_gen)
    test_pred = model.predict(X_test_org_des)

    org_results = pd.DataFrame({'org_train_truelabel':y_train_org_des, 'org_train_pred':org_train_pred})
    gen_results = pd.DataFrame({'gen_train_truelabels':property_gen, 'gen_train_pred':gen_train_pred})
    test_results = pd.DataFrame({'test_truelabels':y_test_org_des, 'test_pred':test_pred})
    #results = pd.concat([org_results, gen_results, test_results], axis=1)
    #results.to_csv(f'prediction_results_{target}.csv', index=False)
    #print('预测数据已保存')
    #return descriptor_gen_norm
    return y_train_org_des, org_train_pred, property_gen, gen_train_pred, y_test_org_des, test_pred

#测试模型
def model_test(target):

    model = joblib.load(f'ML_model_{target}.pkl')

    data_train = pd.read_csv(f'train_data_{target}.csv')
    property_train = data_train.iloc[:, 0]
    feature_train = data_train.iloc[:, 1::]

    data_test = pd.read_csv(f'test_data_{target}.csv')
    property_test = data_test.iloc[:,0]
    feature_test = data_test.iloc[:,1::]

    train_preds = model.predict(feature_train)
    train_R2 = r2_score(property_train, train_preds)
    print(f'训练集R2等于：{train_R2}')
    test_preds = model.predict(feature_test)
    test_R2 = r2_score(property_test, test_preds)
    print(f'测试集R2等于：{test_R2}')
    return train_R2, test_R2

#发现体系
def found_systems(i):

    model_H = joblib.load('ML_model_H.pkl')
    model_M = joblib.load('ML_model_M.pkl')

    # 定义特征名称
    features = ['Cu', 'Al', 'Fe', 'Zr', 'V', 'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta', 'Hf']

    # 生成选择6种特征的所有组合
    combinations_list_7 = list(combinations(features, i))
    weight = 1/i

    # 构造DataFrame，表示每种组合的特征值
    dataframes = []
    for combo in combinations_list_7:
        row = {feature: (weight if feature in combo else 0) for feature in features}
        dataframes.append(row)

    # 转换为DataFrame格式
    data = pd.DataFrame(dataframes)
    # 添加新特征列并赋值为0
    data['pressure'] = 0.5
    data['bias'] = 0.5
    data['flow'] = 0.5

    # 调整列的顺序，确保新列在最前面
    columns_order = ['pressure', 'bias', 'flow'] + [col for col in data.columns if
                                                    col not in ['pressure', 'bias', 'flow']]
    data = data[columns_order]

    composition = data.iloc[:, 3::]
    descriptor_H = compute_descriptor(composition=composition, target='H')
    descriptor_H_norm = Descriptor_norm(descriptor=descriptor_H, target='H')
    descriptor_M = compute_descriptor(composition=composition, target='M')
    descriptor_M_norm = Descriptor_norm(descriptor=descriptor_M, target='M')
    feature_H = pd.concat([data, descriptor_H_norm], axis=1)
    feature_M = pd.concat([data, descriptor_M_norm], axis=1)

    pred_H = model_H.predict(feature_H)
    pred_M = model_M.predict(feature_M)

    H_and_M = pd.DataFrame({'hardness': pred_H, 'modulus': pred_M})
    found_data = pd.concat([H_and_M, data, descriptor_H_norm, descriptor_M_norm], axis=1)
    #found_data.to_csv(f'found_system{i}.csv', index=False)
    #print('体系已保存')

    return found_data

#特征重要性评分
def feature_importance(target):

    model = joblib.load(f'ML_model_{target}.pkl')

    feature_importances = model.feature_importances_

    if target == 'H':
        feature_name = ['pressure', 'bias', 'flow', 'Cu', 'Al', 'Fe', 'Zr', 'V',
                        'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta', 'Hf',
                        'ave:num_f_valence', 'var:evaporation_heat', 'var:num_unfilled', 'var:vdw_radius_alvarez']
    if target == 'M':
        feature_name = ['pressure', 'bias', 'flow', 'Cu', 'Al', 'Fe', 'Zr', 'V',
                        'Co', 'Ni', 'Nb', 'Ti', 'Cr', 'Mo', 'Mn', 'W', 'Ta', 'Hf',
                        'ave:thermal_conductivity', 'var:Polarizability', 'var:gs_energy']

    feature_importances_with_names = list(zip(feature_name, feature_importances))

    feature_importances_sorted = sorted(feature_importances_with_names, key=lambda x: x[1], reverse=True)

    print(f"{target}特征重要性排序：")
    for feature, importance in feature_importances_sorted:
        print(f'{feature}:{importance}')

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

    # 设置字体为新罗马，加粗，字号为18号
    plt.rc('font', family='Times New Roman', weight='bold', size=18)

    # 设置轴和轴标签加粗
    plt.rc('axes', labelweight='bold')

    # 绘制SHAP摘要图
    fig = plt.figure()
    shap.summary_plot(shap_values, data, show=False)  # 禁止自动显示图像

    # 保存图片为高分辨率
    fig.savefig(f'shap_summary_plot_{target}.png', dpi=300, bbox_inches='tight')  # 保存为PNG格式，300 DPI
    plt.close(fig)  # 关闭图像，释放内存

#重新排列列名，用于nsga2
def process_file(file_path):
    # Step 1: Read the file and compute the ratio of the first two columns
    data = pd.read_csv(file_path)

    # Ensure there are at least two columns
    if data.shape[1] < 2:
        raise ValueError("The file must have at least two columns.")

    # Calculate the ratio of the first column to the second column
    data['Ratio'] = data.iloc[:, 0] / data.iloc[:, 1]

    # Step 2: Select samples for specific conditions
    ratio_max = data.loc[data['Ratio'].idxmax()]
    H_max = data.loc[data['hardness'].idxmax()]
    M_max = data.loc[data['modulus'].idxmin()]

    # Combine the three selected rows
    filtered_data = pd.DataFrame([ratio_max, H_max, M_max])

    # Step 3: Rearrange specific column names based on values in each sample
    columns_to_check = ["pressure", "bias", "flow", "Cu", "Al", "Fe", "Zr", "V", "Co", "Ni", "Nb", "Ti", "Cr", "Mo",
                        "Mn", "W", "Ta", "Hf"]
    valid_columns = [col for col in columns_to_check if col in data.columns]

    result = []

    for _, row in filtered_data.iterrows():
        non_zero_cols = [col for col in valid_columns if row[col] != 0]
        zero_cols = [col for col in valid_columns if row[col] == 0]
        rearranged_cols = non_zero_cols + zero_cols
        result.append(rearranged_cols)

    return result

def testdata(data):
    hardness = joblib.load('ML_model_H.pkl')
    modulus = joblib.load('ML_model_M.pkl')

    #工艺归一化
    craft = data[['pressure','bias','flow']]
    #craft = craft_norm(craft)
    #填补工艺缺失值=0.5
    #craft = craft.fillna(0.5)

    #成分归一化
    composition = data[['Cu','Al','Fe','Zr','V','Co','Ni','Nb','Ti','Cr','Mo','Mn','W','Ta','Hf']]
    composition = comp_norm(composition)

    #描述符计算和归一化
    descriotor_H = compute_descriptor(composition, 'H')
    descriotor_M = compute_descriptor(composition, 'M')
    descriotor_H = Descriptor_norm(descriotor_H, 'H')
    descriotor_M = Descriptor_norm(descriotor_M, 'M')

    #特征组合
    feature_H = pd.concat([craft, composition, descriotor_H], axis=1)
    feature_M = pd.concat([craft, composition, descriotor_M], axis=1)

    hardness_pred = hardness.predict(feature_H)
    modulus_pred = modulus.predict(feature_M)
    print(feature_H)
    print(feature_M)

    return hardness_pred, modulus_pred, descriotor_H