# %%

from imodels import RuleFitRegressor
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams
from tqdm import tqdm
import os
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from scipy.stats import pearsonr

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# 读取文件和数据
file_path = r"F:\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data

# %% 删除不需要的列
# 基于事后解释性分析 得到的关键特征列，删除的是不重要的特征
X = data.drop(columns=['CorrRate'])
print(len(X.columns))
print(X.columns)


# 定义 Y
target_name = 'CorrRate'
y = data.loc[:, target_name]
y

# 超参数优化
# 超参数优化
# 超参数优化

def objective(trial):
    # 超参数空间定义
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    tree_size = trial.suggest_int("tree_size", 3, 6)
    max_rules = trial.suggest_int("max_rules", 25, 50, step=5)
    sample_fract = trial.suggest_float("sample_fract", 0.3, 0.7, step=0.1)

    # 初始化RuleFitRegressor模型
    model = RuleFitRegressor(
        n_estimators=n_estimators,
        tree_size=tree_size,
        max_rules=max_rules,
        sample_fract=sample_fract,
        random_state=42
    )

    # 交叉验证
    cv_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 训练模型

        model.fit(X_train_fold, y_train_fold)

        y_val_pred = model.predict(X_val_fold)
        # 计算 RMSE（均方根误差）
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        cv_scores.append(rmse)

    # 返回 RMSE 的平均值
    return np.mean(cv_scores)


# %% 初始化变量和结果存储
best_parameters = []
test_ratio = 0.2
df_result = pd.DataFrame(columns=(
    'RMSE_CV', 'MAE_CV', 'R2_CV', 'Train RMSE', 'Train MAE', 'Train R2', 'Test RMSE', 'Test MAE', 'Test R2'))

# 创建循环 10个模型
for random_seed in range(1, 11, 1):
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # %% 输出特征名
    feature_names = X_train.columns
    feature_names

    # 创建Optuna的study对象
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))  # 改为 minimize，因为RMSE越小越好
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_params['random_state'] = 42

    # 交叉验证最优模型
    # 进行10折交叉验证并显示进度条
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_r2 = []
    cross_val_rmse = []
    cross_val_mae = []
    cross_val_r = []
    # 使用 tqdm 显示每折的进度
    for train_index, val_index in tqdm(kf.split(X_train), total=10, desc="Cross-validation progress"):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # 训练模型
        RuleFit_Optuna_best = RuleFitRegressor(**best_params)
        RuleFit_Optuna_best.fit(X_train_fold, y_train_fold)

        # 预测验证集
        y_val_pred = RuleFit_Optuna_best.predict(X_val_fold)

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

    RuleFit_Optuna_best = RuleFitRegressor(**best_params)
    RuleFit_Optuna_best.fit(X_train, y_train)

    # 汇总10个模型的最佳超参数组合
    best_parameters = ["Best parameters for " + str(random_seed) + ": ", best_params]


    # 训练集进行预测
    y_pred_train = RuleFit_Optuna_best.predict(X_train)

    # 测试集进行预测
    y_pred_test = RuleFit_Optuna_best.predict(X_test)

    # 计算训练集和测试集的相关系数
    train_r, _ = pearsonr(y_train, y_pred_train)
    test_r, _ = pearsonr(y_test, y_pred_test)

    # 计算整个数据集的相关系数
    y_pred_all = RuleFit_Optuna_best.predict(X)
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
    result_dir = './Model_RuleFit'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    df_result.to_excel(f'{result_dir}/Optuna_performance_{random_seed}.xlsx')
    X_test['Observed corrosion_rate'] = y_test
    X_test['Predicted corrosion_rate'] = y_pred_test
    X_test.to_excel(f'{result_dir}/Predict_observe_{random_seed}.xlsx')

    list_str = '\n'.join(str(item) for item in best_parameters)

    # 确保保存最佳参数的文件夹存在
    parameters_dir = './Model_RuleFit'
    if not os.path.exists(parameters_dir):
        os.makedirs(parameters_dir)

    # Write string to file
    with open(f'./Model_RuleFit/Optuna_parameters_{random_seed}.txt', 'w') as file:
        file.write(list_str)

# 提取规则 并保存
    rules = RuleFit_Optuna_best._get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    rules.to_excel("./Model_RuleFit/rules_" + str(random_seed) + ".xlsx")


df_result.describe()