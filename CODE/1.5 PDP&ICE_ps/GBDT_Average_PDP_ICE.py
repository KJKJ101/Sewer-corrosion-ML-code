# %% 导入包
# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import os

from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp
from pdpbox.pdp import PDPIsolate, PDPResult
import lime
import lime.lime_tabular
from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
# 设置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
# 读取数据
file_path = r"E:\Polimi_jiang\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data


# %%
# 数据预处理：去除目标变量列并创建副本
X = data.drop(columns=['CorrRate'])
X_raw = X.copy()  # for SHAP
X_raw


# %%
# 目标变量
target_name = 'CorrRate'
y = data.loc[:, target_name]
y

# %%
# 特征名称
feature_name_list = list(X.columns)
feature_name_list


# 读取超参数配置文件
para_pd = pd.read_excel("./GBDT_parameters.xlsx", index_col=0, )
para_pd


# %% test the models are same with grid-search and obtain ten models
# 10次循环遍历随机种子，根据超参数组合建立10个模型
for random_seed in range(1, 11, 1):

    print('Processing: ', random_seed)
    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    model = GradientBoostingRegressor(random_state=42,

                             n_estimators=para_pd.loc[random_seed, 'n_estimators'],
                             max_depth=para_pd.loc[random_seed, 'max_depth'],
                               learning_rate=para_pd.loc[random_seed, 'learning_rate'],

                               min_samples_split=para_pd.loc[random_seed, 'min_samples_split'],
                               min_samples_leaf=para_pd.loc[random_seed, 'min_samples_leaf'],
                               subsample=para_pd.loc[random_seed, 'subsample'],
                               max_features=para_pd.loc[random_seed, 'max_features'],
                                                           )

    model.fit(X_train, y_train)
    # 根据模型计算 特征重要性
    GBDT_importance_split = model.feature_importances_

    # 测试集进行预测
    y_pred_test = model.predict(X_test)

# 动态创建模型变量
    #根据 random_seed 动态地创建一个变量名（例如 model_1, model_2 等），并将训练好的模型赋值给这个变量。保存每个模型的状态，方便后续调用或保存。
    exec("model_{} = model".format(random_seed))

# 测试集上评估模型性能


    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    print('Test RMSE: %.2f' % rmse)
    mae = metrics.mean_absolute_error(y_test, y_pred_test)
    print('Test MAE: %.2f' % mae)
    r2 = metrics.r2_score(y_test, y_pred_test)
    print('Test R²: %.2f' % r2)


# %%  numerical features for all models
# 生成部分依赖图（PDP）和单个条件的效应（ICE）曲线图


#设置显示 ICE 曲线的比例为 1，意味着展示所有的 ICE 曲线（如果比例设置为其他值，可能只会显示部分）
ratio_show_lines = 1
# 循环遍历每个特征和随机种子
for feature in [ 'H2S',
                ]:

    # 循环遍历每个随机种子并获取模型
    for i in range(1, 11, 1):
        # 通过 exec 获取对应的模型
        exec("GBDT_Optuna_best = model_{}".format(i))

        # 数据集分割，8:2随机划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # 整个数据集中 Location == 1 的全部样本
        X_all_1 = X[X['Location'] == 1]

        # 网格范围取 Location==1 全部数据中该特征的最小/最大值
        # 这样 10 个模型使用同一套网格
        feature_min = X_all_1[feature].min()
        feature_max = X_all_1[feature].max()
        grid_range = (feature_min, feature_max)
        num_grid_points = 20
        X_grid = np.linspace(grid_range[0], grid_range[1], num_grid_points)
        print(f"X_grid for {feature} (from {feature_min} to {feature_max}):")
        print(X_grid)

        # 进行PDP计算
        pdp_NP_none_M = pdp.PDPIsolate(
            model=GBDT_Optuna_best,  # 已训练的回归模型
            df=X_all_1,
            model_features=X_all_1.columns,
            feature=feature,
            feature_name=feature,  # 图表中显示的特征名称
            n_classes=0,
            grid_type='equal',  # 网格类型，可以是 'percentile' 或 'equal'
            num_grid_points=num_grid_points,  # 网格点数
            grid_range=grid_range,  # 网格的最小值和最大值
            n_jobs=-1,  # 使用所有 CPU 核心进行并行计算
        )


        # 从PDPIsolate实例中获取PDPResult
        pdp_result = pdp_NP_none_M.results[0]


        if i == 1:
            ICE_lines = pdp_result.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_result.pdp

        else:
            ICE_lines = pd.concat([ICE_lines, pdp_result.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_result.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_result.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    # 处理缺失值并排序
    # 删除那些非缺失值少于 3 个的列，减少无意义的特征。
    # non_null_counts = PDP_lines.count()
    # PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index]
    # 按字母顺序排序，以确保生成的图像看起来整洁
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1)  # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1)  # sort columns for plot

    # show part of the ice lines
    # 随机选择部分 ICE 曲线并绘图
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    # 绘制 PDP 和 ICE 图并保存
    fig, ax = plt.subplots(figsize=(3, 1.85))
    # 计算 PDP 的均值 和 标准差，即 置信区间
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    # 创建 DataFrame 存储 X_grid 和 pdp_mean
    pdp_data = pd.DataFrame({
        'X_grid': X_grid,
        'pdp_mean': pdp_mean,
        'pdp_std': pdp_std
    })

    # 保存到 Excel 文件
    pdp_data.to_excel(f'./Image/{feature[0:6]}_pdp_data.xlsx', index=False)

    # plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    # 绘制 ICE 曲线
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i, :].dropna()
        # plt.plot(pd_temp.index, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
        # 使用 X_grid 来代替 pd_temp.index 作为 x 轴
        plt.plot(X_grid, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
    # 绘制平均 PDP 曲线和置信区间
    plt.plot(X_grid,pdp_mean, marker='o', markersize=4, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')


    # 修改部分：设置 y 轴的范围为 [0, 5]
    plt.ylim(0.5, 3.5)

    plt.title("PS", fontsize=10, family='Arial')

    # 设置图像标签并保存图像
    plt.xlabel("H2S (ppm)", fontsize=10, family='Arial')
    plt.ylabel('Predicted r', fontsize=10, family='Arial')

    # 确保保存结果的文件夹存在
    result_dir = './Image'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig.savefig("./Image/液相-速率-PDP_ICE_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

    # 绘制和保存简化版本的 PDP 图
    # 用于小尺寸展示
    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    # plt.yticks([0.47,0.50])
    plt.plot(X_grid, pdp_mean, marker='o', markersize=2, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')
    fig.savefig("./Image/液相-速率-PDP_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

#设置显示 ICE 曲线的比例为 1，意味着展示所有的 ICE 曲线（如果比例设置为其他值，可能只会显示部分）
ratio_show_lines = 1
# 循环遍历每个特征和随机种子
for feature in [
                'Temperature']:

# for feature in [ 'Temp.Min (ºC)', ]:
    # 循环遍历每个随机种子并获取模型
    for i in range(1, 11, 1):
        # 通过 exec 获取对应的模型
        exec("GBDT_Optuna_best = model_{}".format(i))

        # 数据集分割，8:2随机划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # 整个数据集中 Location == 1 的全部样本
        X_all_1 = X[X['Location'] == 1]

        # 网格范围取 Location==1 全部数据中该特征的最小/最大值
        # 这样 10 个模型使用同一套网格
        feature_min = X_all_1[feature].min()
        feature_max = X_all_1[feature].max()
        grid_range = (feature_min, feature_max)
        num_grid_points = 20
        X_grid = np.linspace(grid_range[0], grid_range[1], num_grid_points)
        print(f"X_grid for {feature} (from {feature_min} to {feature_max}):")
        print(X_grid)

        # 进行PDP计算
        pdp_NP_none_M = pdp.PDPIsolate(
            model=GBDT_Optuna_best,  # 已训练的回归模型
            df=X_all_1,
            model_features=X_all_1.columns,
            feature=feature,
            feature_name=feature,  # 图表中显示的特征名称
            n_classes=0,
            grid_type='equal',  # 网格类型，可以是 'percentile' 或 'equal'
            num_grid_points=num_grid_points,  # 网格点数
            grid_range=grid_range,  # 网格的最小值和最大值
            n_jobs=-1,  # 使用所有 CPU 核心进行并行计算
        )


        # 从PDPIsolate实例中获取PDPResult
        pdp_result = pdp_NP_none_M.results[0]


        if i == 1:
            ICE_lines = pdp_result.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_result.pdp

        else:
            ICE_lines = pd.concat([ICE_lines, pdp_result.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_result.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_result.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    # 处理缺失值并排序
    # 删除那些非缺失值少于 3 个的列，减少无意义的特征。
    # non_null_counts = PDP_lines.count()
    # PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index]
    # 按字母顺序排序，以确保生成的图像看起来整洁
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1)  # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1)  # sort columns for plot

    # show part of the ice lines
    # 随机选择部分 ICE 曲线并绘图
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    # 绘制 PDP 和 ICE 图并保存
    fig, ax = plt.subplots(figsize=(3, 1.85))
    # 计算 PDP 的均值 和 标准差，即 置信区间
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    # 创建 DataFrame 存储 X_grid 和 pdp_mean
    pdp_data = pd.DataFrame({
        'X_grid': X_grid,
        'pdp_mean': pdp_mean,
        'pdp_std': pdp_std
    })

    # 保存到 Excel 文件
    pdp_data.to_excel(f'./Image/{feature[0:6]}_pdp_data.xlsx', index=False)

    # plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    # 绘制 ICE 曲线
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i, :].dropna()
        # plt.plot(pd_temp.index, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
        # 使用 X_grid 来代替 pd_temp.index 作为 x 轴
        plt.plot(X_grid, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
    # 绘制平均 PDP 曲线和置信区间
    plt.plot(X_grid,pdp_mean, marker='o', markersize=4, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')


    # 修改部分：设置 y 轴的范围为 [0, 5]
    plt.ylim(0.5, 3.5)

    plt.title("PS", fontsize=10, family='Arial')

    # 设置图像标签并保存图像
    plt.xlabel('Temperature (ºC)', fontsize=10, family='Arial')
    plt.ylabel('Predicted r', fontsize=10, family='Arial')

    # 确保保存结果的文件夹存在
    result_dir = './Image'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig.savefig("./Image/液相-速率-PDP_ICE_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

    # 绘制和保存简化版本的 PDP 图
    # 用于小尺寸展示
    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    # plt.yticks([0.47,0.50])
    plt.plot(X_grid, pdp_mean, marker='o', markersize=2, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')
    fig.savefig("./Image/液相-速率-PDP_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

#设置显示 ICE 曲线的比例为 1，意味着展示所有的 ICE 曲线（如果比例设置为其他值，可能只会显示部分）
# H2SAve(mg/m3) 0-8
ratio_show_lines = 1
# 循环遍历每个特征和随机种子
for feature in ['RH'
                ]:

# for feature in [ 'Temp.Min (ºC)', ]:
    # 循环遍历每个随机种子并获取模型
    for i in range(1, 11, 1):
        # 通过 exec 获取对应的模型
        exec("GBDT_Optuna_best = model_{}".format(i))

        # 数据集分割，8:2随机划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # 整个数据集中 Location == 1 的全部样本
        X_all_1 = X[X['Location'] == 1]

        # 网格范围取 Location==1 全部数据中该特征的最小/最大值
        # 这样 10 个模型使用同一套网格
        feature_min = X_all_1[feature].min()
        feature_max = X_all_1[feature].max()
        grid_range = (feature_min, feature_max)
        num_grid_points = 20
        X_grid = np.linspace(grid_range[0], grid_range[1], num_grid_points)
        print(f"X_grid for {feature} (from {feature_min} to {feature_max}):")
        print(X_grid)

        # 进行PDP计算
        pdp_NP_none_M = pdp.PDPIsolate(
            model=GBDT_Optuna_best,  # 已训练的回归模型
            df=X_all_1,
            model_features=X_all_1.columns,
            feature=feature,
            feature_name=feature,  # 图表中显示的特征名称
            n_classes=0,
            grid_type='equal',  # 网格类型，可以是 'percentile' 或 'equal'
            num_grid_points=num_grid_points,  # 网格点数
            grid_range=grid_range,  # 网格的最小值和最大值
            n_jobs=-1,  # 使用所有 CPU 核心进行并行计算
        )


        # 从PDPIsolate实例中获取PDPResult
        pdp_result = pdp_NP_none_M.results[0]


        if i == 1:
            ICE_lines = pdp_result.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_result.pdp

        else:
            ICE_lines = pd.concat([ICE_lines, pdp_result.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_result.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_result.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    # 处理缺失值并排序
    # 删除那些非缺失值少于 3 个的列，减少无意义的特征。
    # non_null_counts = PDP_lines.count()
    # PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index]
    # 按字母顺序排序，以确保生成的图像看起来整洁
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1)  # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1)  # sort columns for plot

    # show part of the ice lines
    # 随机选择部分 ICE 曲线并绘图
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    # 绘制 PDP 和 ICE 图并保存
    fig, ax = plt.subplots(figsize=(3, 1.85))
    # 计算 PDP 的均值 和 标准差，即 置信区间
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    # 创建 DataFrame 存储 X_grid 和 pdp_mean
    pdp_data = pd.DataFrame({
        'X_grid': X_grid,
        'pdp_mean': pdp_mean,
        'pdp_std': pdp_std
    })

    # 保存到 Excel 文件
    pdp_data.to_excel(f'./Image/{feature[0:6]}_pdp_data.xlsx', index=False)

    # plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    # 绘制 ICE 曲线
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i, :].dropna()
        # plt.plot(pd_temp.index, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
        # 使用 X_grid 来代替 pd_temp.index 作为 x 轴
        plt.plot(X_grid, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
    # 绘制平均 PDP 曲线和置信区间
    plt.plot(X_grid,pdp_mean, marker='o', markersize=4, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')


    # 修改部分：设置 y 轴的范围为 [0, 5]
    plt.ylim(0.5, 3.5)

    plt.title("PS", fontsize=10, family='Arial')

    # 设置图像标签并保存图像
    plt.xlabel('RH (%)', fontsize=10, family='Arial')
    plt.ylabel('Predicted r', fontsize=10, family='Arial')

    # 确保保存结果的文件夹存在
    result_dir = './Image'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig.savefig("./Image/液相-速率-PDP_ICE_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

    # 绘制和保存简化版本的 PDP 图
    # 用于小尺寸展示
    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    # plt.yticks([0.47,0.50])
    plt.plot(X_grid, pdp_mean, marker='o', markersize=2, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')
    fig.savefig("./Image/液相-速率-PDP_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')


#设置显示 ICE 曲线的比例为 1，意味着展示所有的 ICE 曲线（如果比例设置为其他值，可能只会显示部分）
# 'Initial pH' 0-6
ratio_show_lines = 1
# 循环遍历每个特征和随机种子
for feature in ['Location']:

# for feature in [ 'Temp.Min (ºC)', ]:
    # 循环遍历每个随机种子并获取模型
    for i in range(1, 11, 1):
        # 通过 exec 获取对应的模型
        exec("GBDT_Optuna_best = model_{}".format(i))

        # 数据集分割，8:2随机划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # 整个数据集中 Location == 1 的全部样本
        X_all_1 = X[X['Location'] == 1]

        # 网格范围取 Location==1 全部数据中该特征的最小/最大值
        # 这样 10 个模型使用同一套网格
        feature_min = X_all_1[feature].min()
        feature_max = X_all_1[feature].max()
        grid_range = (feature_min, feature_max)
        num_grid_points = 20
        X_grid = np.linspace(grid_range[0], grid_range[1], num_grid_points)
        print(f"X_grid for {feature} (from {feature_min} to {feature_max}):")
        print(X_grid)

        # 进行PDP计算
        pdp_NP_none_M = pdp.PDPIsolate(
            model=GBDT_Optuna_best,  # 已训练的回归模型
            df=X_all_1,
            model_features=X_all_1.columns,
            feature=feature,
            feature_name=feature,  # 图表中显示的特征名称
            n_classes=0,
            grid_type='equal',  # 网格类型，可以是 'percentile' 或 'equal'
            num_grid_points=num_grid_points,  # 网格点数
            grid_range=grid_range,  # 网格的最小值和最大值
            n_jobs=-1,  # 使用所有 CPU 核心进行并行计算
        )

        # 从PDPIsolate实例中获取PDPResult
        pdp_result = pdp_NP_none_M.results[0]


        if i == 1:
            ICE_lines = pdp_result.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_result.pdp

        else:
            ICE_lines = pd.concat([ICE_lines, pdp_result.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_result.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_result.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    # 处理缺失值并排序
    # 删除那些非缺失值少于 3 个的列，减少无意义的特征。
    # non_null_counts = PDP_lines.count()
    # PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index]
    # 按字母顺序排序，以确保生成的图像看起来整洁
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1)  # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1)  # sort columns for plot

    # show part of the ice lines
    # 随机选择部分 ICE 曲线并绘图
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    # 绘制 PDP 和 ICE 图并保存
    fig, ax = plt.subplots(figsize=(3, 1.85))
    # 计算 PDP 的均值 和 标准差，即 置信区间
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    # 创建 DataFrame 存储 X_grid 和 pdp_mean
    pdp_data = pd.DataFrame({
        'X_grid': X_grid,
        'pdp_mean': pdp_mean,
        'pdp_std': pdp_std
    })

    # 保存到 Excel 文件
    pdp_data.to_excel(f'./Image/{feature[0:6]}_pdp_data.xlsx', index=False)

    # plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    # 绘制 ICE 曲线
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i, :].dropna()
        # plt.plot(pd_temp.index, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
        # 使用 X_grid 来代替 pd_temp.index 作为 x 轴
        plt.plot(X_grid, pd_temp.values, linewidth=0.1, color='#696969', zorder=-1)
    # 绘制平均 PDP 曲线和置信区间
    plt.plot(X_grid,pdp_mean, marker='o', markersize=4, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')


    # 修改部分：设置 y 轴的范围为 [0, 5]
    plt.ylim(0.5, 3.5)

    plt.title("PS", fontsize=10, family='Arial')

    # 设置图像标签并保存图像
    plt.xlabel("Location", fontsize=10, family='Arial')
    plt.ylabel('Predicted r', fontsize=10, family='Arial')

    # 确保保存结果的文件夹存在
    result_dir = './Image'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fig.savefig("./Image/液相-速率-PDP_ICE_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

    # 绘制和保存简化版本的 PDP 图
    # 用于小尺寸展示
    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    # plt.yticks([0.47,0.50])
    plt.plot(X_grid, pdp_mean, marker='o', markersize=2, label='Average PDP', zorder=2, linewidth=1.5, color='#FF8C00')
    plt.fill_between(X_grid, pdp_mean - pdp_std, pdp_mean + pdp_std, alpha=0.5, zorder=1, color='#FFDAB9')
    fig.savefig("./Image/液相-速率-PDP_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')
