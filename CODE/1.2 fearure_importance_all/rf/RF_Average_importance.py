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
from sklearn.ensemble import RandomForestRegressor
# 设置图中的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# 读取数据
file_path = r"F:\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
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
para_pd = pd.read_excel("./RF_parameters.xlsx", index_col=0, )
para_pd



# 第一段 用于训练模型，评估模型性能
for random_seed in range(1, 11, 1):
    print('Processing: ', random_seed)

    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)


# 构建模型，使用超参数配置文件中的超参数（para_pd）
    model = RandomForestRegressor(random_state=42,
                                  n_estimators=para_pd.loc[random_seed, 'n_estimators'],
                                  max_depth=para_pd.loc[random_seed, 'max_depth'],
                                  min_samples_split=para_pd.loc[random_seed, 'min_samples_split'],

                                  min_samples_leaf=para_pd.loc[random_seed, 'min_samples_leaf'],
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
    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    model = RandomForestRegressor(random_state=42,
                               n_estimators=para_pd.loc[random_seed, 'n_estimators'],
                             max_depth=para_pd.loc[random_seed, 'max_depth'],
                               min_samples_split=para_pd.loc[random_seed, 'min_samples_split'],

                               min_samples_leaf=para_pd.loc[random_seed, 'min_samples_leaf'],
                               max_features=para_pd.loc[random_seed, 'max_features'],

                               )

    model.fit(X_train, y_train)

    # 基于模型计算特征重要性
    XGB_importance_split = model.feature_importances_

    y_pred_test = model.predict(X_test)


# 计算 Permutation Importance，排列特征重要性

    # 用 sklearn 计算 PFI（Permutation Feature Importance）
    result = permutation_importance(model, X_train, y_train, scoring='neg_mean_squared_error',  # 回归使用 MSE
                                    n_repeats=10, random_state=0, n_jobs=-1)
    Permutation_importance = result.importances_mean


    # 计算 SHAP，排列特征重要性
    explainer = shap.TreeExplainer(model=model, data=None, model_output='raw',
                                   feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_train)
    # 计算每个特征的平均绝对shap
    global_shap_values = np.abs(shap_values).mean(0)

    # 保存原始数据
    # 将 XGB_importance_split, Permutation_importance 和 shap_values 保存到文件
    XGB_df = pd.DataFrame(XGB_importance_split, index=X_train.columns, columns=['XGB Importance'])
    PFI_df = pd.DataFrame(Permutation_importance, index=X_train.columns, columns=['PFI Importance'])
    SHAP_df = pd.DataFrame(global_shap_values, index=X_train.columns, columns=['SHAP Importance'])

    # 确保保存路径中的目录存在
    save_dir = './Table/Raw_Importance'
    os.makedirs(save_dir, exist_ok=True)

    XGB_df.to_excel(f'{save_dir}/XGB_Importance_{random_seed}.xlsx')
    PFI_df.to_excel(f'{save_dir}/PFI_Importance_{random_seed}.xlsx')
    SHAP_df.to_excel(f'{save_dir}/SHAP_Importance_{random_seed}.xlsx')

    # 计算相对特征重要性

    # Calculate the relative importance, the maximum is 1
    XGB_importance_split_relative = XGB_importance_split / max(XGB_importance_split)
    Permutation_importance_relative = Permutation_importance / max(Permutation_importance)
    shap_values__relative = global_shap_values / max(global_shap_values)

# 根据相对重要性之和 排序 特征
    # Sort by the sum of relative importance
    # 3种方法 相对重要性 的结果相加 得到综合结果
    importance_sum = XGB_importance_split_relative + Permutation_importance_relative + shap_values__relative
    # 综合重要性进行排序
    sorted_idx_sum = importance_sum.argsort()
    sorted_features = X_train.columns[sorted_idx_sum][::-1]

# 创建并保存相对重要性表格
    importance_relatiove_value = pd.DataFrame(columns=['Feature', 'Relative importance'])
    importance_relatiove_value['Feature'] = X.columns
    # 除以3， 3个指标的平均值
    importance_relatiove_value['Relative importance'] = importance_sum / 3

    # 确保保存路径中的目录存在
    save_dir = './Table'
    os.makedirs(save_dir, exist_ok=True)
    importance_relatiove_value.to_excel('./Table/XGB_Relative importance_' + str(random_seed) + '.xlsx')

# 创建并检查每个特征的详细重要性
    importance_df = pd.DataFrame({'Feature': X_train.columns[sorted_idx_sum],
                                  'GBDT (split)': XGB_importance_split_relative[sorted_idx_sum],
                                  'Permutation': Permutation_importance_relative[sorted_idx_sum],
                                  'SHAP': shap_values__relative[sorted_idx_sum]},
                                 )
# 检查是否有特征未被模型使用
    if importance_df.iloc[0, 1:].sum() == 0:
        print('Warning: At least one feature is not used by the {}th model.'.format(random_seed))
# 将每个特征的排名存储到 importance_rank 中
    for feature in importance_rank.columns:
        index = importance_df.index[importance_df['Feature'] == feature][0]
        importance_rank.loc[random_seed, feature] = len(importance_rank.columns) - index
importance_rank.to_excel('XGB_Importance_Rank.xlsx')
importance_rank

# %%


# %%
if random_seed == 10:
    file_list = os.listdir('./Table')
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

    for i in range(1, 11, 1):
        exec("temp_pd = pd.read_excel(\"./Table/XGB_Relative importance_{}.xlsx\",index_col = 0,)".format(i, i))
        if i == 1:
            Relative_importance_all = pd.DataFrame(columns=temp_pd['Feature'], index=range(1, 11, 1))
            Relative_importance_all.loc[i, :] = temp_pd['Relative importance'].values
        else:
            Relative_importance_all.loc[i, :] = temp_pd['Relative importance'].values

    mean_Relative_importance_all = Relative_importance_all.mean(axis=0)
    mean_Relative_importance_all.sort_values(ascending=False, inplace=True)
    Relative_importance_all = Relative_importance_all[mean_Relative_importance_all.index]

    Relative_importance_all.to_excel('XGB_Importance_Relative.xlsx')
Relative_importance_all

# %%

# %% 检查保存的所有原始数据
if random_seed == 10:
    file_list = os.listdir('./Table/Raw_Importance')
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

    # 初始化 DataFrame 来保存所有文件的数据
    for i in range(1, 11, 1):
        temp_pd_XGB = pd.read_excel(f'./Table/Raw_Importance/XGB_Importance_{i}.xlsx', index_col=0)
        temp_pd_PFI = pd.read_excel(f'./Table/Raw_Importance/PFI_Importance_{i}.xlsx', index_col=0)
        temp_pd_SHAP = pd.read_excel(f'./Table/Raw_Importance/SHAP_Importance_{i}.xlsx', index_col=0)

        # 第一次循环初始化数据框
        if i == 1:
            XGB_all = pd.DataFrame(columns=temp_pd_XGB.index, index=range(1, 11, 1))
            PFI_all = pd.DataFrame(columns=temp_pd_PFI.index, index=range(1, 11, 1))
            SHAP_all = pd.DataFrame(columns=temp_pd_SHAP.index, index=range(1, 11, 1))

        # 将每个文件的数据合并到 DataFrame
        XGB_all.loc[i, :] = temp_pd_XGB.values.flatten()
        PFI_all.loc[i, :] = temp_pd_PFI.values.flatten()
        SHAP_all.loc[i, :] = temp_pd_SHAP.values.flatten()

    # 汇总保存所有原始数据
    XGB_all.to_excel('XGB_Importance_All.xlsx')
    PFI_all.to_excel('PFI_Importance_All.xlsx')
    SHAP_all.to_excel('SHAP_Importance_All.xlsx')

# 返回汇总的 DataFrame
XGB_all, PFI_all, SHAP_all
