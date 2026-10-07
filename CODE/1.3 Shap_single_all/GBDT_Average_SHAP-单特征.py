# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import optimize
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp
import lime
import lime.lime_tabular
from scipy.interpolate import interp1d

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
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
file_path = r"E:\Polimi_jiang\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data

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

# %% check the model performance again
test_ratio = 0.2

for random_seed in range(1, 11, 1):

    print('Processing: ', random_seed)
    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)

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

# # %%
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(
#     X_train)  # each feature contribution is given as a contribution to the probability of the positive class.
#
# # %%
# shap.summary_plot(shap_values, X_train)
#
# # %%
# shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_train)

# %% global average effects/interactions

for random_seed in range(1, 11, 1):
    print('Processing: ', random_seed)
    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    X_test_raw = X_raw.iloc[X_test.index, :]

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

    # 在整个数据集 X 上计算 SHAP（每个种子只建一次 explainer）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_interaction_values = explainer.shap_interaction_values(X)

    if random_seed == 1:
        shap_values_all = shap_values
        shap_interaction_values_all = shap_interaction_values
        X_test_all = X.copy()  # 变量名沿用原来的，方便后面画图代码不改；实际内容是全部数据
    else:
        shap_values_all = np.append(shap_values_all, shap_values, axis=0)
        shap_interaction_values_all = np.concatenate((shap_interaction_values_all, shap_interaction_values), axis=0)
        X_test_all = pd.concat([X_test_all, X])

print('检查')
print(shap_interaction_values_all.shape, shap_values_all.shape, X_test_all.shape, )

# %%
index = feature_name_list.index('Location')
feature_shap_values = shap_values_all[:, index]
print('Average absoulte SHAP value for Location: ', abs(feature_shap_values).mean())

index = feature_name_list.index('H2S')
feature_shap_values = shap_values_all[:, index]
print('Average absoulte SHAP value for H2S: ', abs(feature_shap_values).mean())

index = feature_name_list.index('Temperature')
feature_shap_values = shap_values_all[:, index]
print('Average absoulte SHAP value for Temperature: ', abs(feature_shap_values).mean())

index = feature_name_list.index('RH')
feature_shap_values = shap_values_all[:, index]
print('Average absoulte SHAP value for RH: ', abs(feature_shap_values).mean())

index = feature_name_list.index('Location')
feature_shap_values = shap_interaction_values_all[:, index, index]
print('Average absoulte SHAP interaction value for Location: ', abs(feature_shap_values).mean())

index = feature_name_list.index('H2S')
feature_shap_values = shap_interaction_values_all[:, index, index]
print('Average absoulte SHAP interaction value for H2S: ', abs(feature_shap_values).mean())

index = feature_name_list.index('Temperature')
feature_shap_values = shap_interaction_values_all[:, index, index]
print('Average absoulte SHAP interaction value for Temperature: ', abs(feature_shap_values).mean())

index = feature_name_list.index('RH')
feature_shap_values = shap_interaction_values_all[:, index, index]
print('Average absoulte SHAP interaction value for RH: ', abs(feature_shap_values).mean())

# %% global SHAP values for numerical features
for feature in ['Temperature']:
    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('Temperature (ºC)', fontsize=10, family='Arial')
    plt.ylabel('SHAP value', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_shap_values, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_values_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

for feature in [ 'H2S',]:
    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('H2S (ppm)', fontsize=10, family='Arial')
    plt.ylabel('SHAP value', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_shap_values, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_values_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

for feature in [  'RH',]:
    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('RH (%)', fontsize=10, family='Arial')
    plt.ylabel('SHAP value', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_shap_values, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_values_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

for feature in [  'Location',]:
    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('Location', fontsize=10, family='Arial')
    plt.ylabel('SHAP value', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_shap_values, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_values_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

# SHAP主效应图
# %% global average main effects for numerical features
for feature in ['Temperature']:
    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:, index, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('Temperature (ºC)', fontsize=10, family='Arial')
    plt.ylabel('SHAP main effect', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_main_effects, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_main_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

# %% global average main effects for numerical features
for feature in [ 'H2S']:
    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:, index, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('H2S (ppm)', fontsize=10, family='Arial')
    plt.ylabel('SHAP main effect', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_main_effects, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_main_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

# %% global average main effects for numerical features
for feature in [  'RH',]:
    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:, index, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('RH (%)', fontsize=10, family='Arial')
    plt.ylabel('SHAP main effect', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_main_effects, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_main_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

# %% global average main effects for numerical features
for feature in [  'Location',]:
    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:, index, index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel('Location', fontsize=10, family='Arial')
    plt.ylabel('SHAP main effect', fontsize=10, family='Arial')
    plt.title("Full Data", fontsize=10, family='Arial')
    plt.scatter(X_test_all[feature], global_main_effects, marker='o', s=45, c='#EA8D6C', linewidth=0.4,
                edgecolors='#FFFFFF')
    fig.savefig("./Image/大SHAP_main_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')

# small figures for plot in manuscript
# %% 不显著特征
# feature = 'Temp.Ave (ºC)'
# index = feature_name_list.index(feature)
# global_main_effects = shap_interaction_values_all[:, index, index]
# fig, ax = plt.subplots(figsize=(2, 1.2))
# plt.xlabel(feature)
# plt.ylabel('SHAP main effect')
# plt.scatter(X_test_all[feature], global_main_effects, marker='o', s=30, c='#EA8D6C', linewidth=0.4,
#             edgecolors='#FFFFFF')
# fig.savefig("./Image/小SHAP_main_%s_global.jpg" % feature[0:6], dpi=600, bbox_inches='tight')


# %% SHAP interaction values
# 所有交互的大热点图
# 所有交互的大热点图
# 所有交互的大热点图
fig, ax = plt.subplots(figsize=(8, 8))
plt.style.use('default')
tmp = np.abs(shap_interaction_values_all).sum(0) / 10
for i in range(tmp.shape[0]):
    tmp[i, i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds, :][:, inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

h = sns.heatmap(df_temp2, cmap='flare', square=True, center=12,
                fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False, annot_kws={'size': 12})
bottom, top = ax.get_ylim()

cb = h.figure.colorbar(h.collections[0], shrink=0.85)  # 显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='left', fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=15, rotation_mode='anchor')

fig.savefig("./Image/Global_interactions_all.jpg", dpi=600, bbox_inches='tight')

# 重要交互的热点图
# 重要交互的热点图
# 重要交互的热点图
fig, ax = plt.subplots(figsize=(4, 4))
plt.style.use('default')
tmp = np.abs(shap_interaction_values_all).sum(0) / 10
for i in range(tmp.shape[0]):
    tmp[i, i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds, :][:, inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

df_temp2 = df_temp2.iloc[0:3, 0:3]

h = sns.heatmap(df_temp2, cmap='flare', square=True, center=12,
                fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False, annot_kws={'size': 12})
bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
cb = h.figure.colorbar(h.collections[0], shrink=0.85)  # 显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='left', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Global_interactions_important.jpg", dpi=600, bbox_inches='tight')

