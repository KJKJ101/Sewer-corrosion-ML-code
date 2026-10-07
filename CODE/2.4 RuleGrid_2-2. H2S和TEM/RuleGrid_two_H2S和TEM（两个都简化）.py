# %%
import pandas as pd
import os
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
file_list = os.listdir('./Model_RuleFit')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1, 11, 1):
    exec("temp_pd = pd.read_excel(\"./Model_RuleFit/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('1.RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
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
def count_feature_letters(s):
    letters = ['LOC', 'H2S', 'RH', 'TEM']
    # 使用正则表达式确保更精确的匹配
    import re
    return len(set(re.findall(r'\b(' + '|'.join(map(re.escape, letters)) + r')\b', s)))


# %%
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all = rules_all[rules_all['importance'] > 0]
rules_all

# %%
# 保存过滤后的规则到 Excel 文件：
rules_all.to_excel('2.rules_all_（重要性大于0过滤后的规则）.xlsx')

# %%
rules_two = rules_all[rules_all['Feature number'] == 2]
rules_two
# %%
# 保存过滤后的规则到 Excel 文件：
rules_two.to_excel('3.rules_two（含二个特征的规则）.xlsx')

# %%

H2S_TEM_index = []

for i, rule in enumerate(rules_two['rule']):
    if ('TEM' in rule) and ('H2S' in rule):
        H2S_TEM_index.append(i)

rule_H2S_TEM = rules_two.iloc[H2S_TEM_index, :]
rule_H2S_TEM = rule_H2S_TEM.reset_index(drop=True)

print(rule_H2S_TEM.shape[0])

# %%
rule_H2S_TEM.to_excel('4.rule_H2S_TEM（提取包含目标特征）.xlsx')

# %% the min and max value of each feature
LOC_range = [0, 1]
H2S_range = [0, 50]
RH_range = [90, 100]
TEM_range = [17, 30]

larger_coeff_limit = 0.001
larger_import_limit = 0.001

# %% rule_to_interval_two
def rule_to_interval_two(matches: list, feature_1: str, feature_2: str, feature_1_range: list,
                         feature_2_range: list):  # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'

    if len(matches) == 1:  # if only one statement, so only one feature
        # judge the feature, determin interval based on the logical operator and feature range
        if matches[0][0] == feature_1:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        if matches[0][0] == feature_2:
            if matches[0][1] == '>':
                interval_2 = '(' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_2 = '[' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 2:  # if the rule have two statements
        # judge the number of feature(s)
        if matches[0][0] == matches[1][0]:
            # if only one feature
            if matches[0][0] == feature_1:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
            if matches[0][0] == feature_2:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
                    # if two feature
        else:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

            if matches[1][1] == '>':
                interval_2 = '(' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '>=':
                interval_2 = '[' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ')'
            if matches[1][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ']'

    if len(matches) == 3:
        # judge which feature appear twice
        if matches[0][0] == matches[1][0]:
            # the first feature appear twice
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
                # so the second feature appear once
            if matches[2][1] == '>':
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '>=':
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ')'
            if matches[2][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ']'

        else:
            # the second feature appear twice
            if (matches[1][1] == '>') & (matches[2][1] == '<'):
                interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>') & (matches[2][1] == '<='):
                interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']'
            if (matches[1][1] == '>=') & (matches[2][1] == '<'):
                interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>=') & (matches[2][1] == '<='):
                interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']'
                # so the first feature appear once
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 4:
        # the first feature appear twice
        if (matches[0][1] == '>') & (matches[1][1] == '<'):
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>') & (matches[1][1] == '<='):
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
        if (matches[0][1] == '>=') & (matches[1][1] == '<'):
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>=') & (matches[1][1] == '<='):
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'

        if (matches[2][1] == '>') & (matches[3][1] == '<'):
            interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>') & (matches[3][1] == '<='):
            interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']'
        if (matches[2][1] == '>=') & (matches[3][1] == '<'):
            interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>=') & (matches[3][1] == '<='):
            interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']'

    return interval_1, interval_2
#


# %% simple the intervals for better visualization, if several intervals are close
def simple_interval_func(intervals, threshold=10):
    interval_data = []
    for interval in intervals:
        left_paren, nums, right_paren = interval[0], interval[1:-1], interval[-1]
        num1, num2 = [float(num) for num in nums.split(', ')]
        interval_data.append([left_paren, num1, num2, right_paren])
    interval_df = pd.DataFrame(interval_data, columns=['left_paren', 'num1', 'num2', 'right_paren'])

    interval_df = interval_df.sort_values(by=['num1', 'num2'], ascending=True)
    interval_df['group'] = ''
    interval_df.iloc[0, 4] = 0

    group_start = 0
    for index in range(1, interval_df.shape[0]):
        interval_df_temp = interval_df[interval_df['group'] == group_start]
        if (abs(interval_df.iloc[index, 1] - interval_df_temp['num1'].mean()) <= threshold) and (
                abs(interval_df.iloc[index, 2] - interval_df_temp['num2'].mean()) <= threshold):
            interval_df.iloc[index, 4] = group_start
        else:
            group_start += 1
            interval_df.iloc[index, 4] = group_start

    interval_df['new num1'] = ''
    interval_df['new num2'] = ''

    for group_idnex in range(0, interval_df['group'].max() + 1):
        interval_df_temp = interval_df[interval_df['group'] == group_idnex]
        interval_df.loc[interval_df_temp.index, 'new num1'] = round(interval_df_temp['num1'].mean(), 3)
        interval_df.loc[interval_df_temp.index, 'new num2'] = round(interval_df_temp['num2'].mean(), 3)

    interval_df['simple'] = ''
    for index in interval_df.index:
        interval_df.loc[index, 'simple'] = interval_df.loc[index, 'left_paren'] + str(
            interval_df.loc[index, 'new num1']) + ',' + str(interval_df.loc[index, 'new num2']) + interval_df.loc[
                                               index, 'right_paren']

    interval_df = interval_df.sort_index(axis=0)

    return interval_df['simple'].values

# 第三张图
feature_1 = 'H2S'
feature_2 = 'TEM'

feature_1_range = H2S_range
feature_2_range = TEM_range

rule_H2S_TEM['H2S'] = ''
rule_H2S_TEM['Temperature'] = ''


for i, rule in enumerate(rule_H2S_TEM['rule']):
    matches_raw = re.findall(r'(H2S|TEM)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)

    matches = sorted(matches_raw, key=lambda x: (x[0], float(x[2])))  # sort according to the letter and float number

    rule_H2S_TEM.loc[i, ['H2S', 'Temperature']] = rule_to_interval_two(matches, feature_1,
                                                                                  feature_2,
                                                                                feature_1_range,
                                                                                 feature_2_range)
rule_H2S_TEM['Temperature'].astype(str)
rule_H2S_TEM['H2S'].astype(str)
rule_H2S_TEM

rule_H2S_TEM.to_excel('5 rule_H2S_TEM（转区间后）.xlsx')

# %%
print('rule_H2S_TEM的 TEM 独特区间：')
print(rule_H2S_TEM.sort_values('Temperature')['Temperature'].unique())

rule_H2S_TEM_simply_TEM = pd.DataFrame({'Raw interval': ['(21.0, 27.5]', '(27.5, 30]', '[17, 21.0]', '[17, 23.5]', '[17, 27.5]'] })

#
rule_H2S_TEM_simply_TEM.loc[0:, 'Simply interval'] = simple_interval_func(rule_H2S_TEM_simply_TEM['Raw interval'].values[0:],
                                                                    threshold=(TEM_range[1] - TEM_range[0]) * 0.00000005)
len(rule_H2S_TEM_simply_TEM['Simply interval'].unique())

# %%
rule_H2S_TEM_simply_TEM.to_excel('6.1 H2S_LOC_simply_TEM（简化后的区间）.xlsx')
rule_H2S_TEM_simply_TEM
## 6 文件没问题，以上 合并简化部分 没问题

# replace the interval for better visualization
rule_H2S_TEM_simply = rule_H2S_TEM.copy()
for i, item in enumerate(rule_H2S_TEM_simply['Temperature']):
    index = list(rule_H2S_TEM_simply_TEM['Raw interval']).index(item)
    rule_H2S_TEM_simply.loc[i, 'Temperature'] = rule_H2S_TEM_simply_TEM.loc[index, 'Simply interval']
rule_H2S_TEM_simply
rule_H2S_TEM_simply.to_excel('7.1 H2S_TEM_simply_TEM（简化替换原始）.xlsx')

# %%
print("2.H2S 的独特区间")
print(rule_H2S_TEM.sort_values('H2S')['H2S'].unique())
# %%
rule_H2S_TEM_simply_H2S = pd.DataFrame({'Raw interval': ['(2.5, 37.5]', '(20.0, 50]', '(37.5, 50]', '(7.5, 50]', '[0, 17.5]',
 '[0, 20.0]', '[0, 37.5]', '[0, 7.5]'], })

rule_H2S_TEM_simply_H2S.loc[0:, 'Simply interval'] = simple_interval_func(rule_H2S_TEM_simply_H2S['Raw interval'].values[0:],
                                                                    threshold=(H2S_range[1] - H2S_range[0]) * 0.05)
len(rule_H2S_TEM_simply_H2S['Simply interval'].unique())

# %%
rule_H2S_TEM_simply_H2S.to_excel('6.2 H2S_TEM_simply_H2S（简化后的区间）.xlsx')
rule_H2S_TEM_simply_H2S

# replace the interval for better visualization
# 将原始的间隔换为合并简化后的间隔

for i, item in enumerate(rule_H2S_TEM_simply['H2S']):
    index = list(rule_H2S_TEM_simply_H2S['Raw interval']).index(item)
    rule_H2S_TEM_simply.loc[i, 'H2S'] = rule_H2S_TEM_simply_H2S.loc[index, 'Simply interval']
rule_H2S_TEM_simply
rule_H2S_TEM_simply.to_excel('7.2 H2S_TEM_simply_H2S（简化替换原始）.xlsx')

# %%
rule_map_H2S_TEM_importance = pd.DataFrame(0, index=list(rule_H2S_TEM_simply_TEM['Simply interval'].unique()),
                                       columns=list(rule_H2S_TEM_simply_H2S['Simply interval'].unique()))
rule_map_H2S_TEM_frequency = pd.DataFrame(0, index=list(rule_H2S_TEM_simply_TEM['Simply interval'].unique()),
                                      columns=list(rule_H2S_TEM_simply_H2S['Simply interval'].unique()))
rule_map_H2S_TEM_coefficient = pd.DataFrame(0, index=list(rule_H2S_TEM_simply_TEM['Simply interval'].unique()),
                                        columns=list(rule_H2S_TEM_simply_H2S['Simply interval'].unique()))
rule_map_H2S_TEM_importance

print("检查rule_map_H2S_TEM_importance.index")
print(rule_map_H2S_TEM_importance.index)


# %%
for i in range(rule_H2S_TEM_simply.shape[0]):
    rule_map_H2S_TEM_importance.loc[
        rule_H2S_TEM_simply.loc[i, 'Temperature'], rule_H2S_TEM_simply.loc[i, 'H2S']] += \
    rule_H2S_TEM_simply.loc[i, 'importance'] / 10
    rule_map_H2S_TEM_coefficient.loc[
        rule_H2S_TEM_simply.loc[i, 'Temperature'], rule_H2S_TEM_simply.loc[i, 'H2S']] += \
    rule_H2S_TEM_simply.loc[i, 'coef'] / 10
    rule_map_H2S_TEM_frequency.loc[
        rule_H2S_TEM_simply.loc[i, 'Temperature'], rule_H2S_TEM_simply.loc[i, 'H2S']] += 1 / 10

rule_map_H2S_TEM_importance

print(rule_map_H2S_TEM_importance.index)  # 打印索引，查看区间


# %%
rule_map_H2S_TEM_plot = pd.DataFrame(
    columns=['Temperature', 'H2S', 'Importance', 'Coefficient', 'Frequency'])
for TEM_item in list(rule_H2S_TEM_simply_TEM['Simply interval'].unique()):
    for H2S_item in list(rule_H2S_TEM_simply_H2S['Simply interval'].unique()):
        temp_rule_map = pd.DataFrame({'Temperature': TEM_item,
                                      'H2S': H2S_item,
                                      'Importance': rule_map_H2S_TEM_importance.loc[TEM_item, H2S_item],
                                      'Coefficient': rule_map_H2S_TEM_coefficient.loc[TEM_item, H2S_item],
                                      'Frequency': rule_map_H2S_TEM_frequency.loc[TEM_item, H2S_item]},
                                      index=[0])
        rule_map_H2S_TEM_plot = pd.concat([rule_map_H2S_TEM_plot, temp_rule_map], ignore_index=True)

# 在这里添加四舍五入代码
rule_map_H2S_TEM_plot['Temperature'] = rule_map_H2S_TEM_plot['Temperature'].round(3)
rule_map_H2S_TEM_plot['H2S'] = rule_map_H2S_TEM_plot['H2S'].round(3)
rule_map_H2S_TEM_plot['Importance'] = rule_map_H2S_TEM_plot['Importance'].round(3)
rule_map_H2S_TEM_plot['Coefficient'] = rule_map_H2S_TEM_plot['Coefficient'].round(3)

rule_map_H2S_TEM_plot
rule_map_H2S_TEM_plot.to_excel('8. rule_map_H2S_TEM_plot.xlsx')


# %% 画图
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(2.5, 3))
palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

if max([abs(rule_map_H2S_TEM_coefficient.min().min()), abs(rule_map_H2S_TEM_coefficient.max().max())]) > larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_H2S_TEM_coefficient.min().min()), abs(rule_map_H2S_TEM_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_H2S_TEM_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_H2S_TEM_importance.max().max()
    print('Warning: get larger importance limit for plot!')



# 绘图
g = sns.scatterplot(
    data=rule_map_H2S_TEM_plot, x="Temperature", y="H2S", hue="Coefficient", linewidth=0.2,
    size="Importance", sizes=(20, 250 * rule_map_H2S_TEM_importance.max().max() / larger_import_limit),
    ax=ax, palette=palette, hue_norm=(-larger_coeff_limit, larger_coeff_limit),
)

# 设置x轴和y轴标签
plt.xlabel("Temperature (ºC)", fontsize=12)
plt.ylabel("H2S (ppm)", fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 6.5)

plt.legend(bbox_to_anchor=(0.95, 0.32), loc='upper left',
           borderaxespad=0, labelspacing=0.5, scatterpoints=1,
           frameon=False, ncol=1, handletextpad=0.2
           )

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm, )
sm.set_array([])

# 修改颜色条位置
fig.colorbar(sm, ax=ax, label='Coefficient', shrink=0.7, location='right', pad=0.05, anchor=(0, 1.0))


ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_H2S_TEM.jpg", dpi=600, bbox_inches='tight')