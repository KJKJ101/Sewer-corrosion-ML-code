# %%
import pandas as pd
import os
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# 设置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 读取规则文件
file_list = os.listdir('./Model_RuleFit')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

# 循环10个规则文件，读取 Excel 文件并添加到列表中：
for i in range(1, 11, 1):
    exec("temp_pd = pd.read_excel(\"./Model_RuleFit/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

# 合并所有规则到一个文件中 并重置索引：
rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('1. RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
# 打印总生成规则数量：
print("Total number of generated rules: ", str(rules_all.shape[0]))

# %%
# 定义特征的字典
feature_dict = { 'LOC': 'Location',
                'H2S': 'H2S',
                'RH': 'RH',
                'TEM': 'Temperature',
                }
feature_dict

# %% simplify the rules
# 简化规则：将特征的全名替换为简写：
rules_all['rule'] = rules_all['rule'].str.replace('Location', 'LOC', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('H2S', 'H2S', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('RH', 'RH', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Temperature', 'TEM', regex=False)
rules_all


# %%
# 定义一个函数来计算规则中涉及的特征数：
def count_feature_letters(s):
    letters = ['LOC', 'H2S', 'RH', 'TEM']
    # 使用正则表达式确保更精确的匹配
    import re
    return len(set(re.findall(r'\b(' + '|'.join(map(re.escape, letters)) + r')\b', s)))


# %%
#  应用 count_feature_letters 函数计算特征数，并过滤规则：
# 计算每条规则涉及的特征数量，并过滤掉重要性较小的规则
# 对应第二步
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all = rules_all[rules_all['importance'] > 0.0005]
rules_all

# %%
# 保存过滤后的规则到 Excel 文件：
rules_all.to_excel('2. rules_all_0.1（重要性大于0.0005过滤后的规则）.xlsx')

# %%
rules_one = rules_all[rules_all['Feature number'] == 1]
rules_one = rules_one.round(3)
rules_one
rules_one.to_excel('3. rules_one（只含一个规则）.xlsx')
# %%
# 从规则中提取包含 R 特征的规则

H2S_index = []

for i, rule in enumerate(rules_one['rule']):
    if 'H2S' in rule:
        H2S_index.append(i)

# 根据索引筛选规则并重置索引：
rule_H2S = rules_one.iloc[H2S_index, :]
rule_H2S = rule_H2S.reset_index(drop=True)

print(rule_H2S.shape[0])

# %% the min and max value of each feature
# 定义各特征的最小值和最大值范围：
LOC_range = [0, 1]
H2S_range = [0, 50]
RH_range = [90, 100]
TEM_range = [17, 30]


# can use to set same scale for plot
# give these two variables maxmimum values
larger_coeff_limit = 0.001
larger_import_limit = 0.001


# %% rule_to_interval_one
# 定义规则中的条件转化为区间的函数
# 将规则中的条件（例如 C > 50 或 C >= 25 and C <= 100）转化为区间（例如 (50, 200]）。
def rule_to_interval_one(matches: list, feature_1: str, feature_1_range: list, ):
    interval_1 = 'None'

    if len(matches) == 1:  # if only one statement
        if matches[0][1] == '>':
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
        if matches[0][1] == '>=':
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
        if matches[0][1] == '<':
            interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
        if matches[0][1] == '<=':
            interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 2:  # if the rule has two statements
        if (matches[0][1] == '>') & (matches[1][1] == '<'):
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>') & (matches[1][1] == '<='):
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
        if (matches[0][1] == '>=') & (matches[1][1] == '<'):
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>=') & (matches[1][1] == '<='):
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'

    return interval_1


# %%
# 使用正则表达式解析规则并转换为区间
# 定义特征和范围
feature_1 = 'H2S'
feature_1_range = H2S_range

rule_H2S['H2S'] = ''

for i, rule in enumerate(rule_H2S['rule']):
    matches_raw = re.findall(r'(H2S)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches = sorted(matches_raw, key=lambda x: (x[0], float(x[2])))  # sort according the letter and float number
    rule_H2S.loc[i, ['H2S']] = rule_to_interval_one(matches, feature_1, feature_1_range)

rule_H2S['H2S'].astype(str)
rule_H2S

# 保存到Excel
rule_H2S.to_excel('4. rule_H2S_processed(规则转区间).xlsx', index=False)

# %% simple the intervals for better visualization, if several intervals are close
# 简化一组区间，使得相邻区间非常接近时，它们的中间值能够合并到一个区间中
def simple_interval_func(intervals, threshold=10):
    interval_data = []
    for interval in intervals:
        left_paren, nums, right_paren = interval[0], interval[1:-1], interval[-1]
        num1, num2 = [float(num) for num in nums.split(', ')]
        interval_data.append([left_paren, num1, num2, right_paren])
    interval_df = pd.DataFrame(interval_data, columns=['left_paren', 'num1', 'num2', 'right_paren'])
# 排序区间
    interval_df = interval_df.sort_values(by=['num1', 'num2'], ascending=True)
    # 初始化分组列
    interval_df['group'] = ''
    interval_df.iloc[0, 4] = 0
# 分组相近区间
    group_start = 0
    for index in range(1, interval_df.shape[0]):
        interval_df_temp = interval_df[interval_df['group'] == group_start]
        if (abs(interval_df.iloc[index, 1] - interval_df_temp['num1'].mean()) <= threshold) and (
                abs(interval_df.iloc[index, 2] - interval_df_temp['num2'].mean()) <= threshold):
            interval_df.iloc[index, 4] = group_start
        else:
            group_start += 1
            interval_df.iloc[index, 4] = group_start
# 计算每组区间的平均值
    interval_df['new num1'] = ''
    interval_df['new num2'] = ''

    for group_idnex in range(0, interval_df['group'].max() + 1):
        interval_df_temp = interval_df[interval_df['group'] == group_idnex]
        interval_df.loc[interval_df_temp.index, 'new num1'] = round(interval_df_temp['num1'].mean(), 3)
        interval_df.loc[interval_df_temp.index, 'new num2'] = round(interval_df_temp['num2'].mean(), 3)
# 简化区间
    interval_df['simple'] = ''
    for index in interval_df.index:
        interval_df.loc[index, 'simple'] = interval_df.loc[index, 'left_paren'] + str(
            interval_df.loc[index, 'new num1']) + ',' + str(interval_df.loc[index, 'new num2']) + interval_df.loc[
                                               index, 'right_paren']

    interval_df = interval_df.sort_index(axis=0)

    return interval_df['simple'].values


# %%
print(rule_H2S.sort_values('H2S')['H2S'].unique())

# %%

rule_H2S_simply_H2S = pd.DataFrame({'Raw interval': [
  '(20.0, 50]', '(7.5, 50]', '[0, 20.0]', '[0, 7.5]',

], })
rule_H2S_simply_H2S.loc[0:, 'Simply interval'] = simple_interval_func(rule_H2S_simply_H2S['Raw interval'].values[0:],
                                                                  threshold=(H2S_range[1] - H2S_range[0]) * 0.05)
len(rule_H2S_simply_H2S['Simply interval'].unique())

# %%
# 输出简化后的区间并保存到文件
rule_H2S_simply_H2S.to_excel('5. rule_H2S_simply_H2S（简化合并区间）.xlsx')
rule_H2S_simply_H2S

# %% replace the interval for better visualization
# 替换 Relative weight 列中的原始区间为简化后的区间
rule_H2S_simply = rule_H2S.copy()
for i, item in enumerate(rule_H2S_simply['H2S']):
    index = list(rule_H2S_simply_H2S['Raw interval']).index(item)
    rule_H2S_simply.loc[i, 'H2S'] = rule_H2S_simply_H2S.loc[index, 'Simply interval']
rule_H2S_simply

rule_H2S_simply.to_excel('6. rule_H2S_simply_processed（原始区间替换为简化后的区间）.xlsx', index=False)

# %% 创建用于存储规则映射的数据框
rule_map_H2S_importance = pd.DataFrame(0, index=list(rule_H2S_simply_H2S['Simply interval'].unique()), columns=[''])
rule_map_H2S_frequency = pd.DataFrame(0, index=list(rule_H2S_simply_H2S['Simply interval'].unique()), columns=[''])
rule_map_H2S_coefficient = pd.DataFrame(0, index=list(rule_H2S_simply_H2S['Simply interval'].unique()), columns=[''])
rule_map_H2S_importance

# %% 计算简化区间的统计信息
for i in range(rule_H2S_simply.shape[0]):
    rule_map_H2S_importance.loc[rule_H2S_simply.loc[i, 'H2S'], :] += float(rule_H2S_simply.loc[i, 'importance']) / 10

    rule_map_H2S_coefficient.loc[rule_H2S_simply.loc[i, 'H2S'], :] += float(rule_H2S_simply.loc[i, 'coef']) / 10

    rule_map_H2S_frequency.loc[rule_H2S_simply.loc[i, 'H2S'], :] += 1 / 10


rule_map_H2S_importance

# 自定义格式化函数，将数字保留三位小数
def format_coef(x, pos):
    return f'{x:.3f}'

# %% 画散点图
rule_map_H2S_plot = pd.DataFrame(columns=['H2S', 'y', 'Importance', 'Coefficient', 'Frequency'])
for H2S_item in list(rule_H2S_simply_H2S['Simply interval'].unique()):
    H2S_rule_map = pd.DataFrame({'H2S': H2S_item,
                                  'y': 0,
                                  'Importance': rule_map_H2S_importance.loc[H2S_item, :].values,
                                  'Coefficient': rule_map_H2S_coefficient.loc[H2S_item, :].values,
                                  'Frequency': rule_map_H2S_frequency.loc[H2S_item, :].values,
                                  }, index=[0])

    rule_map_H2S_plot = pd.concat([rule_map_H2S_plot, H2S_rule_map.dropna(axis=1, how='all')], ignore_index=True)

rule_map_H2S_plot

# %% 调整图表的外观，使其更具可读性。
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(2, 1))
palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

if max([abs(rule_map_H2S_coefficient.min().min()), abs(rule_map_H2S_coefficient.max().max())]) > larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_H2S_coefficient.min().min()), abs(rule_map_H2S_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_H2S_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_H2S_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_H2S_plot, x="H2S", y='y', hue="Coefficient", linewidth=0.2,
    size="Importance", sizes=(20, 250 * rule_map_H2S_importance.max().max() / larger_import_limit),
    ax=ax, palette=palette, hue_norm=(-larger_coeff_limit, larger_coeff_limit),
)

plt.xticks(rotation=45, ha='right')
plt.xlabel("H2S (ppm)", fontsize=12)

ax.set_yticks([0])
ax.set_yticklabels('')
ax.set_ylabel('')
plt.xlim(-0.5, 3.5)
plt.ylim(-0.5, 0.5)

plt.legend(bbox_to_anchor=(0.9, 0.08), loc='upper left',
           borderaxespad=0, labelspacing=0.5, scatterpoints=1,
           frameon=False, ncol=1
           )



# 自定义图例中的数值格式
for text in ax.get_legend().get_texts():
    try:
        value = float(text.get_text())
        text.set_text(format_coef(value, 0))  # 将数值格式化为三位小数
    except ValueError:
        pass

# colorbar（依然显示 Coefficient 颜色条）
norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm, )
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient', shrink=0.7, anchor=(0, 0.8), )

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_H2S.jpg", dpi=600, bbox_inches='tight')

# %%
