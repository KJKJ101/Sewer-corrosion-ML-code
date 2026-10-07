# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import shap
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
# 设置图中的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# 读取数据
file_path = r"/Users/kj/Desktop/work_history/Polimi_jiang/ML2024.01/test/dataset/2024.11.23-小数据模型/总体/腐蚀速率/r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data

# 删除目标列并查看特征列
X = data.drop(columns=['CorrRate'])
print(len(X.columns))
print(X.columns)

# 提取目标变量
target_name = 'CorrRate'
y = data.loc[:, target_name]
y

# 读取超参数配置文件
para_pd = pd.read_excel("./GBDT_parameters.xlsx", index_col=0, )
para_pd



# 第一段 用于训练模型，评估模型性能
for random_seed in range(1, 11, 1):
    print('Processing: ', random_seed)

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


# 构建模型，使用超参数配置文件中的超参数（para_pd）
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
    # 测试集进行预测
    y_pred_test = model.predict(X_test)


    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    print('Test RMSE: %.2f' % rmse)
    mae = metrics.mean_absolute_error(y_test, y_pred_test)
    print('Test MAE: %.2f' % mae)
    r2 = metrics.r2_score(y_test, y_pred_test)
    print('Test R²: %.2f' % r2)
    # 计算 R（相关系数）
    # 计算实际值和预测值之间的相关系数
    y_test_mean = np.mean(y_test)
    y_pred_mean = np.mean(y_pred_test)
    covariance = np.sum((y_test - y_test_mean) * (y_pred_test - y_pred_mean)) / len(y_test)
    std_y_test = np.std(y_test)
    std_y_pred = np.std(y_pred_test)
    r = covariance / (std_y_test * std_y_pred)
    print('Test R (相关系数): %.2f' % r)

    # 对整个数据集进行预测
    y_pred_all = model.predict(X)  # X 是整个数据集

    # 计算 RMSE, MAE, R², 和 R
    rmse_all = np.sqrt(metrics.mean_squared_error(y, y_pred_all))
    print('All Data RMSE: %.2f' % rmse_all)

    mae_all = metrics.mean_absolute_error(y, y_pred_all)
    print('All Data MAE: %.2f' % mae_all)

    r2_all = metrics.r2_score(y, y_pred_all)
    print('All Data R²: %.2f' % r2_all)

    # 计算相关系数 R
    y_mean = np.mean(y)
    y_pred_mean = np.mean(y_pred_all)
    covariance_all = np.sum((y - y_mean) * (y_pred_all - y_pred_mean)) / len(y)
    std_y_all = np.std(y)
    std_y_pred_all = np.std(y_pred_all)
    r_all = covariance_all / (std_y_all * std_y_pred_all)

    print('All Data R (相关系数): %.2f' % r_all)

# 第二段 用于计算和保存特征重要性
# %% 创建了一个 空的 DataFrame，
# 它将用来存储不同随机种子下，每个特征的重要性评分。
importance_rank = pd.DataFrame(index=range(1, 11, 1), columns=X.columns)

for random_seed in range(1, 11, 1):

    print('Processing: ', random_seed)
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

    # 模型内置的分裂重要性（由训练过程的树结构决定，不需要额外数据）
    GBDT_importance_split = model.feature_importances_

    # ===== 改动 1：PFI 在测试集上计算 =====
    # 原来：permutation_importance(model, X_train, y_train, ...)
    # PFI 基于预测误差，应使用未参与训练的数据
    # 测试集只有 12-15 个样本，把 n_repeats 从 10 提高到 30 以降低置换本身的随机性
    result = permutation_importance(model, X_test, y_test,
                                    scoring='neg_mean_squared_error',
                                    n_repeats=30, random_state=0, n_jobs=-1)
    Permutation_importance = result.importances_mean

    # ===== 改动 2：SHAP 在全数据集上计算 =====
    # 原来：shap_values = explainer.shap_values(X_train)
    # SHAP 描述模型学到的函数，不涉及观测值，应使用覆盖完整实验设计的全数据集
    explainer = shap.TreeExplainer(model=model, data=None, model_output='raw',
                                   feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X)
    global_shap_values = np.abs(shap_values).mean(0)

    # 以下保持不变
    GBDT_df = pd.DataFrame(GBDT_importance_split, index=X.columns, columns=['GBDT Importance'])
    PFI_df = pd.DataFrame(Permutation_importance, index=X.columns, columns=['PFI Importance'])
    SHAP_df = pd.DataFrame(global_shap_values, index=X.columns, columns=['SHAP Importance'])

    save_dir = './Table/Raw_Importance'
    os.makedirs(save_dir, exist_ok=True)
    GBDT_df.to_excel(f'{save_dir}/GBDT_Importance_{random_seed}.xlsx')
    PFI_df.to_excel(f'{save_dir}/PFI_Importance_{random_seed}.xlsx')
    SHAP_df.to_excel(f'{save_dir}/SHAP_Importance_{random_seed}.xlsx')

    GBDT_importance_split_relative = GBDT_importance_split / max(GBDT_importance_split)
    Permutation_importance_relative = Permutation_importance / max(Permutation_importance)
    shap_values__relative = global_shap_values / max(global_shap_values)

    importance_sum = GBDT_importance_split_relative + Permutation_importance_relative + shap_values__relative
    sorted_idx_sum = importance_sum.argsort()
    sorted_features = X.columns[sorted_idx_sum][::-1]

    importance_relatiove_value = pd.DataFrame(columns=['Feature', 'Relative importance'])
    importance_relatiove_value['Feature'] = X.columns
    importance_relatiove_value['Relative importance'] = importance_sum / 3

    save_dir = './Table'
    os.makedirs(save_dir, exist_ok=True)
    importance_relatiove_value.to_excel('./Table/GBDT_Relative importance_' + str(random_seed) + '.xlsx')

    importance_df = pd.DataFrame({'Feature': X.columns[sorted_idx_sum],
                                  'GBDT (split)': GBDT_importance_split_relative[sorted_idx_sum],
                                  'Permutation': Permutation_importance_relative[sorted_idx_sum],
                                  'SHAP': shap_values__relative[sorted_idx_sum]},
                                 )
    if importance_df.iloc[0, 1:].sum() == 0:
        print('Warning: At least one feature is not used by the {}th model.'.format(random_seed))

    for feature in importance_rank.columns:
        index = importance_df.index[importance_df['Feature'] == feature][0]
        importance_rank.loc[random_seed, feature] = len(importance_rank.columns) - index

importance_rank.to_excel('1. GBDT_Importance_Rank.xlsx')
importance_rank

# %%


# %%
if random_seed == 10:
    file_list = os.listdir('./Table')
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

    for i in range(1, 11, 1):
        exec("temp_pd = pd.read_excel(\"./Table/GBDT_Relative importance_{}.xlsx\",index_col = 0,)".format(i, i))
        if i == 1:
            Relative_importance_all = pd.DataFrame(columns=temp_pd['Feature'], index=range(1, 11, 1))
            Relative_importance_all.loc[i, :] = temp_pd['Relative importance'].values
        else:
            Relative_importance_all.loc[i, :] = temp_pd['Relative importance'].values

    mean_Relative_importance_all = Relative_importance_all.mean(axis=0)
    mean_Relative_importance_all.sort_values(ascending=False, inplace=True)
    Relative_importance_all = Relative_importance_all[mean_Relative_importance_all.index]

    Relative_importance_all.to_excel('2. GBDT_Importance_Relative.xlsx')
Relative_importance_all

# %%

# %% 检查保存的所有原始数据
if random_seed == 10:
    file_list = os.listdir('./Table/Raw_Importance')
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

    # 初始化 DataFrame 来保存所有文件的数据
    for i in range(1, 11, 1):
        temp_pd_GBDT = pd.read_excel(f'./Table/Raw_Importance/GBDT_Importance_{i}.xlsx', index_col=0)
        temp_pd_PFI = pd.read_excel(f'./Table/Raw_Importance/PFI_Importance_{i}.xlsx', index_col=0)
        temp_pd_SHAP = pd.read_excel(f'./Table/Raw_Importance/SHAP_Importance_{i}.xlsx', index_col=0)

        # 第一次循环初始化数据框
        if i == 1:
            GBDT_all = pd.DataFrame(columns=temp_pd_GBDT.index, index=range(1, 11, 1))
            PFI_all = pd.DataFrame(columns=temp_pd_PFI.index, index=range(1, 11, 1))
            SHAP_all = pd.DataFrame(columns=temp_pd_SHAP.index, index=range(1, 11, 1))

        # 将每个文件的数据合并到 DataFrame
        GBDT_all.loc[i, :] = temp_pd_GBDT.values.flatten()
        PFI_all.loc[i, :] = temp_pd_PFI.values.flatten()
        SHAP_all.loc[i, :] = temp_pd_SHAP.values.flatten()

    # 汇总保存所有原始数据
    GBDT_all.to_excel('3.1 速率-GBDT_Importance_All.xlsx')
    PFI_all.to_excel('3.2 速率-PFI_Importance_All.xlsx')
    SHAP_all.to_excel('3.3 速率-SHAP_Importance_All.xlsx')

# 返回汇总的 DataFrame
GBDT_all, PFI_all, SHAP_all
