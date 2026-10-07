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
# from pdpbox import pdp
from pdpbox.pdp import PDPIsolate
import lime
import lime.lime_tabular
from scipy.interpolate import interp1d
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from pdpbox import pdp
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
X


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
    y_pred = model.predict(X_test)


    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    print('Test RMSE: %.2f' % rmse)
    mae = metrics.mean_absolute_error(y_test, y_pred_test)
    print('Test MAE: %.2f' % mae)
    r2 = metrics.r2_score(y_test, y_pred_test)
    print('Test R²: %.2f' % r2)

from mpl_toolkits.mplot3d import Axes3D
all_pdp_results = []

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

X_0 = X[X['Location'] == 0]
all_Z = []
# 循环处理10个模型
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 0
    feature1_max = 50
    feature2_min = 90
    feature2_max = 100

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['H2S', 'RH']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"H2S({v}ppm)" for v in feature1_grid], index=[f"RH({v}%)" for v in feature2_grid])

# 保存为 Excel 文件
# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'H2S & RH _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('H2S (ppm)', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=15, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_H2S & RH_PDP交互.jpg", dpi=800, bbox_inches='tight')

# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('H2S (ppm)', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)
#
# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_H2S & RH_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()

all_Z = []
# 循环处理10个模型
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 0
    feature1_max = 50
    feature2_min = 17
    feature2_max = 30

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['H2S', 'Temperature']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"H2S({v}ppm)" for v in feature1_grid], index=[f"Temperature ({v}℃)" for v in feature2_grid])

# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'H2S & Temp _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('H2S (ppm)', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=16, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=16, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_H2S & Temperature_PDP交互.jpg", dpi=800, bbox_inches='tight')

# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('H2S (ppm)', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)

# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_H2S & Temperature_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()

all_Z = []
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 17
    feature1_max = 30
    feature2_min = 90
    feature2_max = 100

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['Temperature', 'RH']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"Temperature ({v}ºC)" for v in feature1_grid], index=[f"RH ({v}%)" for v in feature2_grid])

# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'TEM & RH _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=16, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=16, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_ Temp & RH_PDP交互.jpg", dpi=800, bbox_inches='tight')

# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)

# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_TEM & RH_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()

all_Z = []
# 循环处理10个模型
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 0
    feature1_max = 1
    feature2_min = 0
    feature2_max = 50

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['Location', 'H2S']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"Location({v})" for v in feature1_grid], index=[f"H2S({v}ppm)" for v in feature2_grid])

# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'LOC & H2S _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('Location', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('H2S (ppm)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=16, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=16, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_Location & H2S_PDP交互.jpg", dpi=800, bbox_inches='tight')


# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('Location', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('H2S (ppm)', fontdict=font_dict, labelpad=15)

# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_LOC & H2S_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()

all_Z = []
# 循环处理10个模型
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 0
    feature1_max = 1
    feature2_min = 90
    feature2_max = 100

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['Location', 'RH']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"Location({v})" for v in feature1_grid], index=[f"RH({v}%)" for v in feature2_grid])

# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'LOC & RH _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('Location', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=16, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=16, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_Location & RH_PDP交互.jpg", dpi=800, bbox_inches='tight')


# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('Location', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('RH (%)', fontdict=font_dict, labelpad=15)

# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_LOC & RH_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()

all_Z = []
# 循环处理10个模型
for i in range(1, 11):
    # 通过 exec 获取对应的模型
    exec("GBDT_Optuna_best = model_{}".format(i))  # 动态获取模型

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # For X_train
    X_train_0 = X_train[X_train['Location'] == 0]
    X_train_1 = X_train[X_train['Location'] == 1]

    # Use the same condition to filter y_train so it matches the corresponding rows of X_train
    y_train_0 = y_train[X_train['Location'] == 0].reset_index(drop=True)
    y_train_1 = y_train[X_train['Location'] == 1].reset_index(drop=True)

    # For X_test
    X_test_0 = X_test[X_test['Location'] == 0]
    X_test_1 = X_test[X_test['Location'] == 1]

    # Use the same condition to filter y_test so it matches the corresponding rows of X_test
    y_test_0 = y_test[X_test['Location'] == 0].reset_index(drop=True)

    def custom_2d_partial_dependence(model, X, features, custom_grids):
        # 创建网格
        grid_x, grid_y = np.meshgrid(custom_grids[0], custom_grids[1])

        # 初始化结果矩阵
        Z = np.zeros_like(grid_x, dtype=float)

        # 遍历网格点
        for i, x_val in enumerate(custom_grids[0]):
            for j, y_val in enumerate(custom_grids[1]):
                X_copy = X.copy()
                X_copy[features[0]] = x_val
                X_copy[features[1]] = y_val
                predictions = model.predict(X_copy)
                Z[j, i] = np.mean(predictions)

        return grid_x, grid_y, Z

    # 手动设置最大最小值
    feature1_min = 0
    feature1_max = 1
    feature2_min = 17
    feature2_max = 30

    # 创建网格
    feature1_grid = np.linspace(feature1_min, feature1_max, 20)
    feature2_grid = np.linspace(feature2_min, feature2_max, 20)

    features = ['Location', 'Temperature']
    custom_grids = [feature1_grid, feature2_grid]

    # 计算部分依赖
    XX, YY, Z = custom_2d_partial_dependence(GBDT_Optuna_best, X_0, features, custom_grids)

    # 将每个模型的PDP结果存储到all_Z中
    all_Z.append(Z)

# 计算所有模型PDP结果的平均值
average_Z = np.mean(all_Z, axis=0)
print(average_Z)

# 保存数据到Excel
# 将数据转换为 DataFrame
df = pd.DataFrame(average_Z, columns=[f"Location({v})" for v in feature1_grid], index=[f"Temperature({v})" for v in feature2_grid])

# 获取当前 Python 文件所在的目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 构建保存 Excel 文件的路径
output_file = os.path.join(current_dir, 'LOC & TEM _pdp_results.xlsx')

# 保存为 Excel 文件
df.to_excel(output_file)

# 绘制最终的3D PDP图
# 统一的字体和字号设置
font_dict = {'fontsize': 15, 'family': 'Arial', 'color': 'black'}

# 绘制最终的3D PDP图
# 绘制3D PDP图
fig = plt.figure(figsize=(6.4, 4.8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(XX, YY, average_Z, cmap='Purples', edgecolor='k', alpha=0.8)

# 设置标签和标题
ax.set_xlabel('Location', fontdict=font_dict, labelpad=15)  # 调整x轴标题位置
ax.set_ylabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)  # 调整y轴标题位置


# 设置刻度标签的字体大小和颜色
ax.tick_params(axis='x', labelsize=15, labelcolor='black')  # 设置x轴刻度
ax.tick_params(axis='y', labelsize=15, labelcolor='black')  # 设置y轴刻度
ax.tick_params(axis='z', labelsize=15, labelcolor='black')  # 设置z轴刻度

# # 添加颜色条并设置标题
# cbar = plt.colorbar(surf, shrink=0.8, aspect=10)  # 删除fontsize参数
# cbar.set_label('Predicted r', fontsize=16, color='black')  # 为颜色条添加标题并设置字体大小
# cbar.ax.tick_params(labelsize=16, labelcolor='black')

# 调整视角
ax.view_init(elev=22, azim=122)

plt.tight_layout()


# 确保保存结果的文件夹存在
result_dir = './Image_PDP交互'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 保存3D图
fig.savefig("./Image_PDP交互/PDP_Location & Temperature _PDP交互.jpg", dpi=800, bbox_inches='tight')


# 绘制热图
fig_heatmap, ax_heatmap = plt.subplots(figsize=(6.4, 4.8))

# 使用指定的颜色进行绘制
cmap = plt.get_cmap("Purples")

# 绘制热图
heatmap = ax_heatmap.imshow(average_Z, aspect='auto', cmap=cmap, origin='lower', extent=[feature1_min, feature1_max, feature2_min, feature2_max])

# 设置标签和标题
ax_heatmap.set_xlabel('Location', fontdict=font_dict, labelpad=15)
ax_heatmap.set_ylabel('Temperature (ºC)', fontdict=font_dict, labelpad=15)

# 设置图的标题
ax_heatmap.set_title('GP', fontdict=font_dict, pad=10)

# 添加颜色条并设置标题
cbar_heatmap = fig_heatmap.colorbar(heatmap, ax=ax_heatmap, shrink=0.8, aspect=10)
cbar_heatmap.set_label('Predicted r', fontsize=15, color='black')  # 为颜色条添加标题并设置字体大小
cbar_heatmap.ax.tick_params(labelsize=15, labelcolor='black')

# 调整热图样式
ax_heatmap.tick_params(axis='both', labelsize=15, labelcolor='black')  # 设置刻度字体大小

# 保存热图
heatmap_result_dir = './Image_PDP_Heatmap'
if not os.path.exists(heatmap_result_dir):
    os.makedirs(heatmap_result_dir)

# 保存热图
fig_heatmap.savefig("./Image_PDP_Heatmap/PDP_LOC & TEM_Heatmap.jpg", dpi=800, bbox_inches='tight')
plt.tight_layout()