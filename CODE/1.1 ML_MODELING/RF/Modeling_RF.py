# %% 导入包


import matplotlib.pyplot as plt
from matplotlib import rcParams


from tqdm import tqdm
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
from scipy.stats import pearsonr

seed = 42
#设置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# 读取文件和数据
file_path = r"F:\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data

# 绘图S4
# 包含多个子图（subplot）的图表，每个子图显示一个变量的可视化。
plt.figure(figsize=(8, 10))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
# 绘图的位置调整
for i in range(1, 5, 1):
    # 绘图 4行 3列 共12个
    plt.subplot(4, 3, i)
    # 绘图 1 7 10 是 条形图，否则是直方图
    # 涉及到 类别 特征 就是条形图，连续的数字变量 就是直方图
    if i in [120, 121, 122]:
        if i == 1:
            data[data.columns[i - 1]].value_counts().plot(kind='bar', color="#FF8C00",
                                                          edgecolor="black", alpha=0.7, width=1)
            plt.xticks(rotation=45, horizontalalignment='right')
        elif i == 7:
            data[data.columns[i - 1]].value_counts().plot(kind='bar', color="#FF8C00",
                                                          edgecolor="black", alpha=0.7, width=0.45)
            plt.xticks(rotation=0, horizontalalignment='center')
        else:
            data[data.columns[i - 1]].value_counts().plot(kind='bar', color="#FF8C00",
                                                          edgecolor="black", alpha=0.7, width=0.15)
            plt.xticks(rotation=0, horizontalalignment='center')
    else:
        plt.hist(data.iloc[:, i - 1], facecolor="#FF8C00", edgecolor="black", alpha=0.7)
    plt.xlabel(data.columns[i - 1])
    plt.ylabel("Freqency")

output_dir = "./RF_Image"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
plt.savefig("./RF_Image/气相_dataset_visual.jpg", dpi=600, bbox_inches='tight')

# 定义 X 和 Y
# 去除 corrosion_rate（即 预测标签） 后的 新X
X = data.drop(columns=['CorrRate'])
print(len(X.columns))
print(X.columns)


# 定义 Y
target_name = 'CorrRate'
y = data.loc[:, target_name]
y



# 构建 RF 模型 和 Optuna 超参数优化

def objective(trial):
    # Optuna 超参数搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_int("max_features", 1, len(X.columns)),
    }

    # 创建 RandomForest 回归模型
    model = RandomForestRegressor(
        **params,
        random_state=42,

    )

    # 交叉验证评估
    cv_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        # 计算 RMSE（均方根误差）
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        cv_scores.append(rmse)

    # 返回 RMSE 的平均值
    return np.mean(cv_scores)


best_parameters = []
test_ratio = 0.2
df_result = pd.DataFrame(columns=(
    'RMSE_CV', 'MAE_CV', 'R2_CV', 'Train RMSE', 'Train MAE', 'Train R2', 'Test RMSE', 'Test MAE', 'Test R2'))

# 10个随机种子，循环10次，建立10个模型
for random_seed in range(1, 11, 1):
    # 数据集分割，8:2随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Optuna 优化
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))  # 改为 minimize，因为RMSE越小越好
    study.optimize(objective, n_trials=300)

    # 使用最佳超参数创建模型
    best_params = study.best_params
    best_params['random_state'] = 42


    # 交叉验证最优模型
    # 进行10折交叉验证并显示进度条
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_r2 = []
    cross_val_rmse = []
    cross_val_mae = []
    cross_val_r = []  # 用于存储每折的相关系数

    # 使用 tqdm 显示每折的进度
    for train_index, val_index in tqdm(kf.split(X_train), total=10, desc="Cross-validation progress"):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # 训练模型
        RF_Optuna_best = RandomForestRegressor(**best_params)
        RF_Optuna_best.fit(X_train_fold, y_train_fold)

        # 预测验证集
        y_val_pred = RF_Optuna_best.predict(X_val_fold)

        # 计算 R² 和 RMSE
        fold_r2 = r2_score(y_val_fold, y_val_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        fold_mae = mean_absolute_error(y_val_fold, y_val_pred)


        # 计算相关系数
        fold_r, _ = pearsonr(y_val_fold, y_val_pred)

        cross_val_r2.append(fold_r2)
        cross_val_rmse.append(fold_rmse)
        cross_val_mae.append(fold_mae)
        cross_val_r.append(fold_r)

    # 计算10折交叉验证的平均值
    mean_r2_cv = np.mean(cross_val_r2)
    mean_rmse_cv = np.mean(cross_val_rmse)
    mean_mae_cv = np.mean(cross_val_mae)
    mean_r_cv = np.mean(cross_val_r)  # 平均相关系数

    RF_Optuna_best = RandomForestRegressor(**best_params)
    RF_Optuna_best.fit(X_train, y_train)

    # 汇总10个模型的最佳超参数组合
    best_parameters = ["Best parameters for " + str(random_seed) + ": ", best_params]

    # 训练集进行预测
    y_pred_train = RF_Optuna_best.predict(X_train)

    # 测试集进行预测
    y_pred_test = RF_Optuna_best.predict(X_test)

    # 计算训练集和测试集的相关系数
    train_r, _ = pearsonr(y_train, y_pred_train)
    test_r, _ = pearsonr(y_test, y_pred_test)

    # 计算整个数据集的相关系数
    y_pred_all = RF_Optuna_best.predict(X)
    all_r, _ = pearsonr(y, y_pred_all)

    # 添加新的行到 df_result
    result_row = {
        'RMSE_CV': mean_rmse_cv,
        'MAE_CV': mean_mae_cv,
        'R2_CV': mean_r2_cv,
        'Correlation_Coefficient_CV': mean_r_cv,  # 添加交叉验证相关系数
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Train R2': r2_score(y_train, y_pred_train),
        'Train Correlation': train_r,  # 训练集相关系数
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Test R2': r2_score(y_test, y_pred_test),
        'Test Correlation': test_r,  # 测试集相关系数
        'Overall RMSE': np.sqrt(mean_squared_error(y, y_pred_all)),
        'Overall MAE': mean_absolute_error(y, y_pred_all),
        'Overall R2': r2_score(y, y_pred_all),
        'Overall Correlation': all_r  # 整个数据集的相关系数
    }

    # 将结果作为 DataFrame 添加到 df_result
    df_result = pd.concat([df_result, pd.DataFrame([result_row])], ignore_index=True)

    # 确保保存结果的文件夹存在
    result_dir = './Model_RF'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    df_result.to_excel(f'{result_dir}/Optuna_performance_{random_seed}.xlsx')
    X_test['Observed corrosion_rate'] = y_test
    X_test['Predicted corrosion_rate'] = y_pred_test
    X_test.to_excel(f'{result_dir}/Predict_observe_{random_seed}.xlsx')

    list_str = '\n'.join(str(item) for item in best_parameters)

    # 确保保存最佳参数的文件夹存在
    parameters_dir = './Model_RF'
    if not os.path.exists(parameters_dir):
        os.makedirs(parameters_dir)

    # Write string to file
    with open(f'./Model_RF/Optuna_parameters_{random_seed}.txt', 'w') as file:
        file.write(list_str)

df_result.describe()

# %%