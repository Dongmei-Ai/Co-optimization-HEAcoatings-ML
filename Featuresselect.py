import numpy as np
import pandas as pd
import itertools
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from xenonpy.descriptor import Compositions
from scipy.stats import spearmanr
from xenonpy.datatools import preset


# 成分归一化  输入：composition  输出：normalized_composition
def comp_norm(composition):
    normalized_composition = composition.div(composition.sum(axis=1), axis=0)
    return normalized_composition

# 特征归一化  输入：feature  输出：normalized_feature
def feature_norm(feature):

    # 初始化MinMaxScaler
    scaler = MinMaxScaler()
    # 对所有列进行归一化
    normalized_feature = pd.DataFrame(scaler.fit_transform(feature), columns=feature.columns)
    return normalized_feature

# 描述符计算  输入：composition  输出：df_descriptor
def compute_descriptor(composition):

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
    df_descriptor = df_descriptor.drop(['sum:hhi_r','sum:hhi_p','ave:hhi_r','ave:hhi_p','var:hhi_p','var:hhi_r'], axis=1)

    return df_descriptor


# 数据集加载  输入：path 输出：property, craft_norm, composition_norm, descriptor_norm
def data_load(path,i=0,j=4):
    data = pd.read_csv(path)
    property = data.iloc[:,i]
    craft = data.iloc[:,i+1:j]
    composition = data.iloc[:,j:]

    # 成分归一化
    composition = comp_norm(composition)
    # 工艺归一化
    craft =feature_norm(craft)

    # 计算描述符
    descriptor = compute_descriptor(composition)
    # 描述符归一化
    descriptor = feature_norm(descriptor)
    return property, craft, composition, descriptor

# 特征选择1  输入：descriptor  输出：descriptor_different（方差不等于0的）
def select_features_frist(descriptor):
    descriptor_different = []
    # 删除方差为零的列
    for column in descriptor.columns:
        if descriptor[column].max() != descriptor[column].min():
            descriptor_different.append(column)
    descriptor_different = descriptor[descriptor_different]

    return descriptor_different

# 特征选择2  输入：descriptor, target_variable（性能）  输出：descriptor（斯皮尔曼相关系数大且相互独立的）
def select_features_second(descriptor, target_variable):
    # 计算特征之间的 Spearman 相关系数
    matrix_correlation=  pd.DataFrame(index=descriptor.columns, columns=descriptor.columns)
    for col1 in descriptor.columns:
        for col2 in descriptor.columns:
            correlation, _ = spearmanr(descriptor[col1], descriptor[col2])
            matrix_correlation.loc[col1, col2] = correlation

    descriptor_delay = []
    # 计算特征与目标变量之间的相关系数，并选择相关性较高的特征
    for col1 in matrix_correlation.columns:
        for col2 in matrix_correlation.columns:
            if col1 < col2 and abs(matrix_correlation.loc[col1, col2]) > 0.9:  # 仅处理上三角矩阵的一半
                correlation_with_target1, _ = spearmanr(descriptor[col1], target_variable)
                correlation_with_target2, _ = spearmanr(descriptor[col2], target_variable)

                # 保存相关性较小的特征到 descriptor_delay 属性中
                if correlation_with_target1 < correlation_with_target2:
                    descriptor_delay.append(col1)
                else:
                    descriptor_delay.append(col2)

    # 删除 descriptor_delay 中的特征，并保存到 descriptor_keep 属性中
    descriptor_keep = list(set(descriptor.columns) - set(descriptor_delay))
    descriptor = descriptor[descriptor_keep]
    return descriptor

# 特征选择3  输入：X（工艺-成分-描述符）, y（性能）  输出：selected_features
def select_features_third(model, X, y):
    # 初始化随机森林回归模型
    np.random.seed(42)
    #rf = RandomForestRegressor(random_state=42)
    model = model

    # 使用递归特征消除（RFE）进行特征选择
    #selector = RFE(estimator=rf, step=1, n_features_to_select=10)
    selector = RFE(estimator=model, step=1, n_features_to_select=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    # 返回选择的特征
    return selected_features

# 特征选择4  输入：property,craft,composition,descriptor  输出：方差不等于0-相关系数大-递归特征消除后的descriptor
def select_features_fourth(model, property,craft,composition,descriptor):
    descriptor_frist = select_features_frist(descriptor)  # descriptor_frist中不包括方差等于0的特征
    descriptor_second = select_features_second(descriptor_frist, property) # descriptor_second仅包括特征之间相关系数＜0.9且与property相关系数大的特征
    # 拼接工艺、成分和两步特征选择后的描述符
    X = pd.concat([craft, composition, descriptor_second], axis=1)
    feature = select_features_third(model=model, X=X, y=property)  # feature可能不包含某些工艺和成分特征
    columns_to_remove = set(craft.columns).union(set(composition.columns))  # 获取craft和composition的列名

    descriptor_third = feature.difference(columns_to_remove)  # 删除feature中的这些列名

    return descriptor[descriptor_third]

# 数据集划分  输入：feature，property  输出：X_train, X_test, y_train, y_test
def split_data(feature, property, seed=42):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(feature, property,test_size=0.2,
                                                        random_state=seed)
    X_train = X_train.reset_index(drop=True)  #重置索引
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test

models = {
        "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "BaggingRegressor": BaggingRegressor(random_state=42),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(random_state=42)}

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

# 特征选择5  输入：X_train, X_test, y_train, y_test, descriptor(需要遍历的特征子集)
def select_features_fifth(model, X_train, X_test, y_train, y_test, descriptor):

    X_train_craft_comp = X_train.iloc[::,:18]
    X_test_craft_comp = X_test.iloc[::, :18]

    # 评估函数
    def evaluate_feature_subset(model, X_train, y_train, X_test, y_test):
        #model = RandomForestRegressor(random_state=42)
        model = model
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return scores.mean(), r2_score(y_pred, y_test)

    # 获取所有特征子集的名称
    descriptor_name = descriptor.columns

    # 初始化最佳分数和最佳特征组合
    best_r2 = -np.inf
    best_features = None
    best_test_r2 = None
    # 遍历所有特征组合
    for r in range(1, len(descriptor_name) + 1):
        for subset in itertools.combinations(descriptor_name, r):
            subset_list = list(subset)  # 将 tuple 转换为 list
            X_train_subset_add = pd.concat([X_train_craft_comp, X_train[subset_list]], axis=1)
            X_test_subset_add = pd.concat([X_test_craft_comp, X_test[subset_list]], axis=1)
            current_r2, test_r2 = evaluate_feature_subset(model=model, X_train=X_train_subset_add, y_train=y_train, X_test=X_test_subset_add, y_test=y_test)
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_features = subset
                best_test_r2 = test_r2
    return list(best_features), best_r2, best_test_r2

def follow(path, model):
    property, craft, composition, descriptor = data_load(path, i=0, j=4)
    # 方差不等于0 - 相关系数大 - 递归特征消除后的descriptor
    descriptor1 = select_features_fourth(model=model, property=property, craft=craft, composition=composition, descriptor=descriptor)
    # 数据集划分
    feature = pd.concat([craft, composition, descriptor1], axis=1)
    X_train, X_test, y_train, y_test = split_data(feature=feature, property=property)
    # 穷举得到的描述符的名字
    descriptor2, best_r2, test_r2_sf = select_features_fifth(model,X_train, X_test, y_train, y_test, descriptor1)
    X_train_afterselect = pd.concat([X_train.iloc[::, :18], X_train[descriptor2]], axis=1)
    X_test_afterselect = pd.concat([X_test.iloc[::, :18], X_test[descriptor2]], axis=1)
    train_r2, test_r2 = train_model(X_train_afterselect, X_test_afterselect, y_train, y_test)
    return descriptor2, best_r2, test_r2_sf, train_r2, test_r2, X_train

if __name__=="__main__":
    for name, model in models.items():
        if name == 'BaggingRegressor':
            continue
        descriptor2, best_r2, test_r2_sf, train_r2, test_r2, X_train = follow(path='data_H.csv', model=model)
        print('以{}为选择的模型'.format(name))
        print('    最佳描述符组合：{}'.format(descriptor2))
        print('    训练集最优r2：{}, 相应测试集r2:{}'.format(best_r2, test_r2_sf))
        print('    ML模型在最优描述符组合下训练集r2：{}\n    ML模型在最优描述符组合下训练集r2:{}'.format(train_r2, test_r2))