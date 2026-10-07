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

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(
    X_train)  # each feature contribution is given as a contribution to the probability of the positive class.



# %%
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_train)

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

    # 整个数据集中 Location == 1 的全部样本
    X_all_1 = X[X['Location'] == 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_all_1)
    shap_interaction_values = explainer.shap_interaction_values(X_all_1)

    if random_seed == 1:
        shap_values_all = shap_values
        shap_interaction_values_all = shap_interaction_values
        X_test_all = X_all_1.copy()   # 变量名沿用，后面画图代码不用改
    else:
        shap_values_all = np.append(shap_values_all, shap_values, axis=0)
        shap_interaction_values_all = np.concatenate((shap_interaction_values_all, shap_interaction_values), axis=0)
        X_test_all = pd.concat([X_test_all, X_all_1])

print('检查')
print(shap_interaction_values_all.shape, shap_values_all.shape, X_test_all.shape, )

# %%
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


# %% global average interactions
# 1.BC交互
feature_1 = 'Location'
feature_2 = 'RH'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('Location', fontsize=10, family='Arial')


plt.title("PS", fontsize=10, family='Arial')

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax, label="RH (%)")

fig.savefig("./Image/液相-速率-SHAP_inter_Location & RH_global.jpg", dpi=600, bbox_inches='tight')

# %% global average interactions
# 2.BR交互
feature_1 = 'Location'
feature_2 = 'H2S'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects


fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('Location', fontsize=10, family='Arial')

plt.title("PS", fontsize=10, family='Arial')

# Remove the legend and add a colorbar
ax.get_legend().remove()

ax.figure.colorbar(sm, ax=ax, label="H2S (ppm)")

fig.savefig("./Image/液相-速率-SHAP_inter_Location & H2S_global.jpg", dpi=600, bbox_inches='tight')

# %% global average interactions
# 3.RC交互
feature_1 = 'Location'
feature_2 = 'Temperature'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('Location', fontsize=10, family='Arial')

plt.title("PS", fontsize=10, family='Arial')

ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax, label='Temperature (ºC)')

fig.savefig("./Image/液相-速率-SHAP_inter_Location & TEM_global.jpg", dpi=600, bbox_inches='tight')

feature_1 = 'RH'
feature_2 = 'Temperature'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('RH (%)', fontsize=10, family='Arial')

plt.title("PS", fontsize=10, family='Arial')

ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax, label='Temperature (ºC)')

fig.savefig("./Image/液相-速率-SHAP_inter_RH & TEM_global.jpg", dpi=600, bbox_inches='tight')

feature_1 = 'H2S'
feature_2 = 'Temperature'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('H2S (ppm)', fontsize=10, family='Arial')

plt.title("PS", fontsize=10, family='Arial')

ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax, label='Temperature (ºC)')

fig.savefig("./Image/液相-速率-SHAP_inter_H2S & TEM_global.jpg", dpi=600, bbox_inches='tight')


feature_1 = 'H2S'
feature_2 = 'RH'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:, index_1, index_2]

data_plot = pd.DataFrame(columns=[feature_1, feature_2, 'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions', hue=feature_2, data=data_plot,
                palette='flare', s=45)


norm = plt.Normalize(vmin=data_plot[feature_2].min(), vmax=data_plot[feature_2].max())

sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])
plt.xlabel('H2S (ppm)', fontsize=10, family='Arial')

plt.title("PS", fontsize=10, family='Arial')

ax.get_legend().remove()
ax.figure.colorbar(sm, ax=ax, label='RH (%)')

fig.savefig("./Image/液相-速率-SHAP_inter_H2S & RH_global.jpg", dpi=600, bbox_inches='tight')
# %%