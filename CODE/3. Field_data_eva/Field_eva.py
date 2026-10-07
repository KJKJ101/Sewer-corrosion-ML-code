# -*- coding: utf-8 -*-
"""
RuleFit 野外数据验证（请在 myenv 环境中运行）

训练方式与 RuleFit 建模代码完全一致：
  X = data.drop(columns=[target])       # 列顺序与原训练一致
  train_test_split(X, y, test_size=0.2, random_state=seed), seed = 1..10
  RuleFitRegressor(**Optuna_parameters_{seed}.txt 中的最优参数)，在 80% 训练集上训练

运行时会用原来保存的 Optuna_performance_{seed}.xlsx 核对测试集指标，确认复现成功。

输出格式与 GBDT 验证脚本一致：
  RuleFit-{任务}-野外验证-长表.xlsx   —— 给画图代码用
  RuleFit-{任务}-野外验证-指标.xlsx   —— 逐模型指标 / 逐样本汇总 / 测试集复现核对 / 运行信息
"""

import os
import re
import ast

import numpy as np
import pandas as pd
import sklearn
from importlib.metadata import version as pkg_version
from imodels import RuleFitRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ============================================================
# 路径配置
# ============================================================

ROOT = r"E:\Polimi_jiang\ML2024.01\pythonProject1\2026.03-可解释性下水道腐蚀\2025.1.13-可解释性实操（正确数据）"

r_train_path  = ROOT + r"\数据集\r.xlsx"
ti_train_path = ROOT + r"\数据集\Ti.xlsx"

r_field_path  = ROOT + r"\实操\野外验证\r_yewai.xlsx"
ti_field_path = ROOT + r"\实操\野外验证\ti_yewai.xlsx"

r_model_dir  = ROOT + r"\实操\r\2.1 Rulefit_建模（基于myenv自建环境）\Model_RuleFit"
ti_model_dir = ROOT + r"\实操\ti\2.1 Rulefit_建模（基于myenv自建环境）\Model_RuleFit"

OUTPUT_DIR = ROOT + r"\实操\野外验证\Field_Validation_Results_RuleFit"

TEST_RATIO = 0.2
SEEDS = range(1, 11)


# ============================================================
# 工具函数
# ============================================================

def load_params(model_dir, seed):
    path = os.path.join(model_dir, f"Optuna_parameters_{seed}.txt")
    with open(path, 'r') as f:
        content = f.read()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    params = ast.literal_eval(match.group())
    params['random_state'] = 42
    return params


def load_saved_test_metrics(model_dir, seed):
    """读取原训练时保存的测试集指标（文件是累积的，取最后一行即为该 seed）。"""
    path = os.path.join(model_dir, f"Optuna_performance_{seed}.xlsx")
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path, index_col=0)
    last = df.iloc[-1]
    return {'Test RMSE': last.get('Test RMSE', np.nan),
            'Test R2': last.get('Test R2', np.nan)}


def calc_metrics(y_obs, y_pred, train_mean):
    """
    主指标：NRMSE / NMAE 以本次评估集（野外）实测均值归一化。
    带 train-mean 的列以训练集均值归一化，仅供对照。
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    mae = mean_absolute_error(y_obs, y_pred)

    if len(y_obs) >= 3 and np.std(y_obs) > 0 and np.std(y_pred) > 0:
        r, p = pearsonr(y_obs, y_pred)
    else:
        r, p = np.nan, np.nan

    obs_mean = y_obs.mean()
    return {
        'R': r,
        'p_value': p,
        'RMSE': rmse,
        'MAE': mae,
        'NRMSE (%)': rmse / obs_mean * 100,
        'NMAE (%)': mae / obs_mean * 100,
        'NRMSE_train-mean (%)': rmse / train_mean * 100,
        'NMAE_train-mean (%)': mae / train_mean * 100,
    }


# ============================================================
# 单个任务
# ============================================================

def run_rulefit_field_validation(train_path, field_path, model_dir, target_col,
                                 obs_col, pred_col, task_name):
    print(f"\n{'=' * 60}\n  RuleFit - {task_name} 野外数据验证\n{'=' * 60}")

    # 与原训练代码一致
    data = pd.read_excel(train_path, index_col=0)
    X = data.drop(columns=[target_col])
    y = data[target_col].astype(float)
    feature_order = list(X.columns)
    print(f"  训练数据: {len(data)} 样本, 特征顺序 = {feature_order}")

    df_field = pd.read_excel(field_path, index_col=0)
    X_field = df_field[feature_order]
    y_field = df_field[target_col].astype(float)
    print(f"  野外样本量: {len(df_field)}")

    long_rows, metric_rows, check_rows, preds = [], [], [], {}

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_RATIO, random_state=seed)

        params = load_params(model_dir, seed)
        model = RuleFitRegressor(**params)
        model.fit(X_train, y_train)

        # ---------- 复现核对：测试集指标 ----------
        y_pred_test = model.predict(X_test)
        rmse_now = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_now = r2_score(y_test, y_pred_test)
        saved = load_saved_test_metrics(model_dir, seed)
        if saved is not None:
            match = bool(np.isclose(rmse_now, saved['Test RMSE'], rtol=1e-4) and
                         np.isclose(r2_now, saved['Test R2'], rtol=1e-4))
        else:
            match = None
        check_rows.append({'Model': f'Model {seed}',
                           'Test RMSE (now)': rmse_now,
                           'Test RMSE (saved)': saved['Test RMSE'] if saved else np.nan,
                           'Test R2 (now)': r2_now,
                           'Test R2 (saved)': saved['Test R2'] if saved else np.nan,
                           'Reproduced': match})

        # ---------- 野外预测 ----------
        y_pred = model.predict(X_field)
        preds[seed] = y_pred

        m = calc_metrics(y_field, y_pred, y_train.mean())
        metric_rows.append({'Model': f'Model {seed}',
                            'Train_size': len(X_train),
                            'Train_mean': y_train.mean(),
                            **m})

        for i, sid in enumerate(df_field.index):
            row = {'Model': seed, 'Sample_index': sid}
            row.update(X_field.iloc[i].to_dict())
            row[obs_col] = y_field.iloc[i]
            row[pred_col] = y_pred[i]
            long_rows.append(row)

        if match is True:
            flag = ''
        elif match is None:
            flag = '   <-- 未找到原测试集结果文件'
        else:
            flag = (f"   <-- 与原测试集不一致: RMSE {rmse_now:.4f} vs {saved['Test RMSE']:.4f}, "
                    f"R2 {r2_now:.4f} vs {saved['Test R2']:.4f}")
        print(f"  Model {seed:2d}: 测试集复现 {match} | 野外 R = {m['R']:.4f}, "
              f"NRMSE = {m['NRMSE (%)']:.2f}%, NMAE = {m['NMAE (%)']:.2f}%{flag}")

    # ---------- 长表 ----------
    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(OUTPUT_DIR, f"RuleFit-{task_name}-野外验证-长表.xlsx")
    long_df.to_excel(long_path, index=False)

    # ---------- 逐模型指标 + 汇总 ----------
    metric_df = pd.DataFrame(metric_rows)
    num_cols = ['R', 'RMSE', 'MAE', 'NRMSE (%)', 'NMAE (%)',
                'NRMSE_train-mean (%)', 'NMAE_train-mean (%)']
    mean_row = {'Model': 'Mean of 10 models', **metric_df[num_cols].mean().to_dict()}
    std_row = {'Model': 'Std of 10 models (ddof=1)', **metric_df[num_cols].std(ddof=1).to_dict()}

    pred_mat = np.column_stack([preds[s] for s in SEEDS])
    ens_pred = pred_mat.mean(axis=1)
    ens_row = {'Model': 'Ensemble mean prediction',
               **calc_metrics(y_field, ens_pred, y.mean())}

    metric_out = pd.concat([metric_df, pd.DataFrame([mean_row, std_row, ens_row])],
                           ignore_index=True)

    # ---------- 逐样本汇总 ----------
    per_sample = X_field.copy()
    per_sample[obs_col] = y_field
    for s in SEEDS:
        per_sample[f'Pred Model {s}'] = preds[s]
    per_sample['Pred_mean'] = ens_pred
    per_sample['Pred_std (ddof=1)'] = pred_mat.std(axis=1, ddof=1)
    per_sample['Pred_min'] = pred_mat.min(axis=1)
    per_sample['Pred_max'] = pred_mat.max(axis=1)
    per_sample['Rel_error_of_mean (%)'] = np.abs(ens_pred - y_field) / y_field * 100

    # ---------- 运行信息 ----------
    info = pd.DataFrame({
        'Item': ['Model', 'Training', 'Seeds', 'Main normalization',
                 'Field observed mean', 'Training data mean (all)',
                 'imodels', 'scikit-learn', 'numpy', 'pandas'],
        'Value': ['RuleFitRegressor (imodels)',
                  'train_test_split(test_size=0.2, random_state=seed), fit on 80%',
                  f'{min(SEEDS)}-{max(SEEDS)}',
                  'NRMSE / NMAE divided by mean of field observed values',
                  y_field.mean(), y.mean(),
                  pkg_version('imodels'), sklearn.__version__, np.__version__, pd.__version__],
    })

    metric_path = os.path.join(OUTPUT_DIR, f"RuleFit-{task_name}-野外验证-指标.xlsx")
    with pd.ExcelWriter(metric_path) as writer:
        metric_out.to_excel(writer, sheet_name='逐模型指标', index=False)
        per_sample.to_excel(writer, sheet_name='逐样本汇总')
        pd.DataFrame(check_rows).to_excel(writer, sheet_name='测试集复现核对', index=False)
        info.to_excel(writer, sheet_name='运行信息', index=False)

    print(f"\n  10 模型平均: R = {mean_row['R']:.4f}, "
          f"NRMSE = {mean_row['NRMSE (%)']:.2f}%, NMAE = {mean_row['NMAE (%)']:.2f}%")
    print(f"  Ensemble  : R = {ens_row['R']:.4f}, "
          f"NRMSE = {ens_row['NRMSE (%)']:.2f}%, NMAE = {ens_row['NMAE (%)']:.2f}%")
    print("\n  逐样本（实测 vs 10 模型预测均值 [最小, 最大]）:")
    for sid in per_sample.index:
        r_ = per_sample.loc[sid]
        print(f"    样本 {sid}: {r_[obs_col]:.3f} vs {r_['Pred_mean']:.3f} "
              f"[{r_['Pred_min']:.3f}, {r_['Pred_max']:.3f}]")
    print(f"\n  长表已保存: {long_path}")
    print(f"  指标已保存: {metric_path}")


# ============================================================
# 运行
# ============================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_rulefit_field_validation(
        train_path=r_train_path,
        field_path=r_field_path,
        model_dir=r_model_dir,
        target_col='CorrRate',
        obs_col='Observed corrosion_rate',
        pred_col='Predicted corrosion_rate',
        task_name='速率',
    )

    run_rulefit_field_validation(
        train_path=ti_train_path,
        field_path=ti_field_path,
        model_dir=ti_model_dir,
        target_col='Tin',
        obs_col='Observed initiation_time',
        pred_col='Predicted initiation_time',
        task_name='起始时间',
    )

    print(f"\n{'=' * 60}\n  全部完成\n{'=' * 60}")