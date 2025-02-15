import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from xenonpy.datatools import preset
from xenonpy.descriptor import Compositions
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, Concatenate, Activation,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


tf.random.set_seed(0)

def CGAN(craft, composition, property, save_path, HorM):
    # 构建 cGAN 模型
    def build_cgan_model(property, craft, composition, output_g=18):

        # 生成器网络
        # 性能-工艺-成分
        g1_input = Input(shape=(property,))
        g2_input = Input(shape=(craft,))
        g3_input = Input(shape=(composition,))
        g_concat = Concatenate()([g1_input, g2_input, g3_input])
        g_h1 = Dense(16)(g_concat)
        g_h1 = BatchNormalization()(g_h1)
        g_h1 = Activation('relu')(g_h1)

        g_h2 = Dense(16)(g_h1)
        g_h2 = BatchNormalization()(g_h2)
        g_h2 = Activation('relu')(g_h2)

        g_h3 = Dense(64)(g_h2)
        g_h3 = BatchNormalization()(g_h3)
        g_h3 = Activation('relu')(g_h3)
        g_h3 = Dropout(0.2)(g_h3)

        g_h4 = Dense(64)(g_h3)  # 90
        g_h4 = BatchNormalization()(g_h4)
        g_h4 = Activation('relu')(g_h4)
        g_h4 = Dropout(0.2)(g_h4)

        # 输出层：27个节点，采用tanh激活函数
        g_output = Dense(output_g, activation='sigmoid')(g_h4)

        generator = Model([g1_input, g2_input, g3_input], g_output)

        # 判别器网络
        d_input = Input(shape=(property + craft + composition,))
        d_h1 = Dense(64)(d_input)
        d_h1 = LeakyReLU(alpha=0.2)(d_h1)
        d_h1 = BatchNormalization()(d_h1)
        d_h1 = Dropout(0.2)(d_h1)

        # 第二隐藏层：100个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h2 = Dense(128)(d_h1)
        d_h2 = LeakyReLU(alpha=0.2)(d_h2)
        d_h2 = BatchNormalization()(d_h2)
        d_h2 = Dropout(0.2)(d_h2)

        # 第三隐藏层：50个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h3 = Dense(64)(d_h2)
        d_h3 = LeakyReLU(alpha=0.2)(d_h3)
        d_h3 = BatchNormalization()(d_h3)
        d_h3 = Dropout(0.2)(d_h3)

        # 第四隐藏层：32个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h4 = Dense(32)(d_h3)
        d_h4 = LeakyReLU(alpha=0.2)(d_h4)
        d_h4 = BatchNormalization()(d_h4)
        d_h4 = Dropout(0.2)(d_h4)

        # 输出层：1个节点，采用sigmoid激活函数
        d_output = Dense(1, activation='sigmoid')(d_h4)

        discriminator = Model(d_input, d_output)

        # 编译生成器和判别器
        #generator.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0005, beta_1=0.5))
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
                              metrics=['accuracy'])

        # 冻结判别器，构建 cGAN
        discriminator.trainable = False
        cgan_output = discriminator(
            Concatenate()([g1_input, generator([g1_input, g2_input, g3_input])]))
        cgan = Model([g1_input, g2_input, g3_input], cgan_output)
        cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        return generator, discriminator, cgan

    # 训练 cGAN 模型
    def train_cgan(generator, discriminator, cgan, craft, composition, property, HorM=HorM, epochs=5000, batch_size=32):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 记录损失的列表
        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            idx = np.random.randint(0, craft.shape[0], batch_size)
            real_craft = craft.iloc[idx]
            real_composition = composition.iloc[idx]
            real_property = property[idx]
            real_property = np.expand_dims(real_property, axis=1)  # 调整维度

            noise_craft = np.random.normal(0, 1, (batch_size, 3))
            noise_composition = np.random.normal(0, 1, (batch_size, 15))
            generated_features = generator.predict([real_property, noise_craft, noise_composition])

            d_loss_real = discriminator.train_on_batch(
                np.concatenate([real_property, real_craft, real_composition], axis=1), valid)

            d_loss_fake = discriminator.train_on_batch(np.concatenate([real_property, generated_features], axis=1),
                                                       fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise_craft = np.random.normal(0, 1, (batch_size, 3))
            noise_composition = np.random.normal(0, 1, (batch_size, 15))
            g_loss = cgan.train_on_batch([real_property, noise_craft, noise_composition], valid)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            # 将损失数据保存到 DataFrame 中
            loss_data = pd.DataFrame({'Discriminator Loss': d_losses, 'Generator Loss': g_losses})
            loss_data.to_csv(f'{HorM}_CGAN_loss.csv', index=False)

    generator, discriminator, cgan = build_cgan_model(craft=3, composition=15, property=1)

    # 训练 cGAN 模型
    train_cgan(generator, discriminator, cgan, craft=craft, composition=composition, property=property, HorM=HorM)

    # 保存生成器模型
    generator.save(save_path)

def DHG_CGAN(craft, composition, property, save_path, HorM):
    # 构建 cGAN 模型
    def build_cgan_model(craft, composition, property):

        # 生成器网络性能-工艺-成分
        # 工艺网络
        g1_input = Input(shape=(craft,))
        g1_h1 = Dense(16)(g1_input)
        g1_h1 = BatchNormalization()(g1_h1)
        g1_h1 = Activation('relu')(g1_h1)

        g1_h2 = Dense(16)(g1_h1)
        g1_h2 = BatchNormalization()(g1_h2)
        g1_h2 = Activation('relu')(g1_h2)

        # 成分网络
        g2_input = Input(shape=(composition,))
        g2_h1 = Dense(16)(g2_input)
        g2_h1 = BatchNormalization()(g2_h1)
        g2_h1 = Activation('relu')(g2_h1)

        g2_h2 = Dense(16)(g2_h1)
        g2_h2 = BatchNormalization()(g2_h2)
        g2_h2 = Activation('relu')(g2_h2)

        g_c = Input(shape=(property,))
        g_concat = Concatenate()([g_c, g1_h2, g2_h2])
        g3_h1 = Dense(64)(g_concat)  # 90
        g3_h1 = BatchNormalization()(g3_h1)
        g3_h1 = Activation('relu')(g3_h1)
        g3_h1 = Dropout(0.2)(g3_h1)

        # 第二隐藏层：200个节点，采用relu激活函数和批次归一化，使用dropout
        g3_h2 = Dense(64)(g3_h1)  # 90
        g3_h2 = BatchNormalization()(g3_h2)
        g3_h2 = Activation('relu')(g3_h2)
        g3_h2 = Dropout(0.2)(g3_h2)

        # 输出层：27个节点，采用tanh激活函数
        g_output = Dense(18, activation='sigmoid')(g3_h2)

        generator = Model([g_c, g1_input, g2_input], g_output)

        # 判别器网络
        d_input = Input(shape=(craft + composition + property,))
        d_h1 = Dense(64)(d_input)
        d_h1 = LeakyReLU(alpha=0.2)(d_h1)
        d_h1 = BatchNormalization()(d_h1)
        d_h1 = Dropout(0.2)(d_h1)

        # 第二隐藏层：100个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h2 = Dense(128)(d_h1)
        d_h2 = LeakyReLU(alpha=0.2)(d_h2)
        d_h2 = BatchNormalization()(d_h2)
        d_h2 = Dropout(0.2)(d_h2)

        # 第三隐藏层：50个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h3 = Dense(64)(d_h2)  # 45
        d_h3 = LeakyReLU(alpha=0.2)(d_h3)
        d_h3 = BatchNormalization()(d_h3)
        d_h3 = Dropout(0.2)(d_h3)

        # 第四隐藏层：32个节点，采用LeakyReLU激活函数，进行批次归一化和dropout
        d_h4 = Dense(32)(d_h3)  # 45
        d_h4 = LeakyReLU(alpha=0.2)(d_h4)
        d_h4 = BatchNormalization()(d_h4)
        d_h4 = Dropout(0.2)(d_h4)

        # 输出层：1个节点，采用sigmoid激活函数
        d_output = Dense(1, activation='sigmoid')(d_h4)

        discriminator = Model(d_input, d_output)

        # 编译生成器和判别器
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
                              metrics=['accuracy'])

        # 冻结判别器，构建 cGAN
        discriminator.trainable = False
        cgan_output = discriminator(
            Concatenate()([g_c, generator([g_c, g1_input, g2_input])]))
        cgan = Model([g_c, g1_input, g2_input], cgan_output)
        cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        return generator, discriminator, cgan

    # 训练 cGAN 模型
    def train_cgan(generator, discriminator, cgan, craft, composition, property, HorM=HorM, epochs=5000, batch_size=32):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 记录损失的列表
        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            idx = np.random.randint(0, craft.shape[0], batch_size)
            real_craft = craft.iloc[idx]
            real_composition = composition.iloc[idx]
            real_property = property[idx]
            real_property = np.expand_dims(real_property, axis=1)  # 调整维度

            noise_craft = np.random.normal(0, 1, (batch_size, 3))
            noise_composition = np.random.normal(0, 1, (batch_size, 15))
            generated_features = generator.predict([real_property, noise_craft, noise_composition])

            d_loss_real = discriminator.train_on_batch(
                np.concatenate([real_property, real_craft, real_composition], axis=1), valid)

            d_loss_fake = discriminator.train_on_batch(np.concatenate([real_property, generated_features], axis=1),
                                                       fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise_craft = np.random.normal(0, 1, (batch_size, 3))
            noise_composition = np.random.normal(0, 1, (batch_size, 15))
            g_loss = cgan.train_on_batch([real_property, noise_craft, noise_composition], valid)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            # 将损失数据保存到 DataFrame 中
            loss_data = pd.DataFrame({'Discriminator Loss': d_losses, 'Generator Loss': g_losses})
            loss_data.to_csv(f'{HorM}_DHGCGANloss.csv', index=False)

    generator, discriminator, cgan = build_cgan_model(craft=3, composition=15, property=1)

    # 训练 cGAN 模型
    train_cgan(generator, discriminator, cgan, craft=craft, composition=composition, property=property, HorM=HorM)

    # 保存生成器模型
    generator.save(save_path)

# 成分归一化  输入：composition  输出：normalized_composition
def comp_norm(composition):
    normalized_composition = composition.div(composition.sum(axis=1), axis=0)
    return normalized_composition

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

# 生成数据成分归一化和描述符计算并选择  输入：描述符名字，成分  输出：标准化成分，描述符（特征选择后）
def augemented_normalized(composition, descriptor_name):
    normalized_composition = comp_norm(composition)  # 成分归一化
    generated_descriptor = compute_descriptor(normalized_composition)  #计算描述符
    generated_descriptor_afterselection = generated_descriptor[descriptor_name]  #描述符选择
    return normalized_composition, generated_descriptor_afterselection

# 生成数据标准化  输入：生成数据， 描述符  输出：
def generateddata_norm(gen_df, descriptor):
    gen_pro_craft = gen_df.iloc[::,:4]
    gen_comp = gen_df.iloc[::,4:]
    gen_comp_norm, gen_descrip = augemented_normalized(descriptor_name=descriptor.columns, composition=gen_comp)
    generateddata_norm = pd.DataFrame(np.hstack((gen_pro_craft, gen_comp_norm, gen_descrip)), columns=np.concatenate((gen_df.columns,descriptor.columns)))
    return generateddata_norm

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

# 特征归一化  输入：feature  输出：normalized_feature
def feature_norm(feature):

    # 初始化MinMaxScaler
    scaler = MinMaxScaler()
    # 对所有列进行归一化
    normalized_feature = pd.DataFrame(scaler.fit_transform(feature), columns=feature.columns)
    return normalized_feature

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

def follow(path, descrip_name, property_low, property_high, HorM):
    # 导入数据集-计算描述符-归一化
    property, craft, composition, descriptor = data_load(path, i=0, j=4)
    # 最优描述符组合
    descriptor = descriptor[descrip_name]
    features = pd.concat([craft, composition, descriptor], axis=1)
    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(feature=features, property=property)
    train_craft = X_train.iloc[::, :3]
    train_composition = X_train.iloc[::, 3:18]
    train_descriptor = X_train.iloc[::, 18:]
    # 训练生成模型
    CGAN(craft=train_craft, composition=train_composition, property=y_train, save_path=f'{HorM}_cgan.keras', HorM=HorM)
    DHG_CGAN(craft=train_craft, composition=train_composition, property=y_train, save_path=f'{HorM}_dhgcgan.keras', HorM=HorM)

    # 生成数据
    # 导入模型
    generator_cgan = load_model(f'{HorM}_cgan.keras')
    generator_dhgcgan = load_model(f'{HorM}_dhgcgan.keras')

    property_craft_comp_name = pd.concat([y_train, train_craft, train_composition], axis=1).columns
    for sample in range(50, 301, 50):
        noise_craft = np.random.normal(0, 1, (sample, 3))
        noise_composition = np.random.normal(0, 1, (sample, 15))
        noise_property = np.random.normal(0, 1, (sample, 1))

        # CGAN数据生成
        generated_conditions = np.random.uniform(low=property_low, high=property_high, size=(sample, 1))
        generated_data_cgan = generator_cgan.predict([generated_conditions, noise_craft, noise_composition])
        generated_data_cgan = pd.DataFrame(np.hstack((generated_conditions, generated_data_cgan)),
                                           columns=property_craft_comp_name)

        # DHGCGAN数据生成
        generated_data_dhgcgan = generator_dhgcgan.predict([generated_conditions, noise_craft, noise_composition])
        generated_data_dhgcgan = pd.DataFrame(np.hstack((generated_conditions, generated_data_dhgcgan)),
                                              columns=property_craft_comp_name)

        # 虚拟数据标准化
        gen_data_cgan_norm = generateddata_norm(generated_data_cgan, train_descriptor)
        gen_data_dhgcgan_norm = generateddata_norm(generated_data_dhgcgan, train_descriptor)

        # 数据集保存
        gen_data_cgan_norm.to_csv(f'{HorM}_cgan_generated_data_{sample}.csv', index=False)
        gen_data_dhgcgan_norm.to_csv(f'{HorM}_dhgcgan_generated_data_{sample}.csv', index=False)
    return X_train

X_train_H = follow(path='data_H.csv', descrip_name=['ave:num_f_valence','var:evaporation_heat','var:num_unfilled','var:vdw_radius_alvarez'],property_low=10,
       property_high=30, HorM='H')

X_train_M = follow(path='data_M.csv', descrip_name=['ave:thermal_conductivity','var:Polarizability','var:gs_energy'],property_low=100,
       property_high=300, HorM='M')

