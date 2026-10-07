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
rules_all.to_excel('1. RuleFit_all_rules（总规则汇总）.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ", str(rules_all.shape[0]))

# %%
feature_dict = { 'LOC': 'Location',
                'H2S': 'H2S',
                'RH': 'RH',
                'TEM': 'Temperature',
                }
feature_dict

# %% simplify the rules
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
# 保存过滤后的规则到 Excel 文件：
rules_all.to_excel('2.rules_all_0.1（重要性大于0过滤后的规则）.xlsx')

# %%
rules_three = rules_all[rules_all['Feature number'] == 3]
rules_three
# 保存过滤后的规则到 Excel 文件：
rules_three.to_excel('3.rules_three（含三个特征的规则）.xlsx')

# %%
H2S_LOC_TEM_index = []


for i, rule in enumerate(rules_three['rule']):
    if ('H2S' in rule) and ('LOC' in rule) and ('TEM' in rule):
        H2S_LOC_TEM_index.append(i)

rule_H2S_LOC_TEM = rules_three.iloc[H2S_LOC_TEM_index, :]
rule_H2S_LOC_TEM = rule_H2S_LOC_TEM.reset_index(drop=True)


print('rule_H2S_LOC_TEM的规则数量：')
print(rule_H2S_LOC_TEM.shape[0])
rule_H2S_LOC_TEM.to_excel('4.1 rule_H2S_LOC_TEM（提取包含目标特征）.xlsx')


# %% the min and max value of each feature
# %% the min and max value of each feature

LOC_range = [0, 1]
H2S_range = [0, 50]
RH_range = [90, 100]
TEM_range = [17, 30]



# set to 0 when first run, then change to the max value by manual

larger_coeff_limit = 0.001
larger_import_limit = 0.001


# %% rule_to_interval_three
def rule_to_interval_three(matches: list, feature_1: str, feature_2: str, feature_3: str, feature_1_range: list,
                           feature_2_range: list, feature_3_range: list):  # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'
    interval_3 = 'None'

    feature_in = [matches[i][0] for i in range(len(matches))]

    feature_1_count = feature_in.count(feature_1)
    feature_2_count = feature_in.count(feature_2)
    feature_3_count = feature_in.count(feature_3)

    start = 0

    if feature_1_count > 0:
        if feature_1_count == 1:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        else:
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'
        start += feature_1_count

    if feature_2_count > 0:
        if feature_2_count == 1:
            if matches[start + 0][1] == '>':
                interval_2 = '(' + str(matches[start + 0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start + 0][1] == '>=':
                interval_2 = '[' + str(matches[start + 0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start + 0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start + 0][2]) + ')'
            if matches[start + 0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start + 0][2]) + ']'
        else:
            if (matches[start + 0][1] == '>') & (matches[start + 1][1] == '<'):
                interval_2 = '(' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ')'
            if (matches[start + 0][1] == '>') & (matches[start + 1][1] == '<='):
                interval_2 = '(' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ']'
            if (matches[start + 0][1] == '>=') & (matches[start + 1][1] == '<'):
                interval_2 = '[' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ')'
            if (matches[start + 0][1] == '>=') & (matches[start + 1][1] == '<='):
                interval_2 = '[' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ']'
        start += feature_2_count

    if feature_3_count > 0:
        if feature_3_count == 1:
            if matches[start + 0][1] == '>':
                interval_3 = '(' + str(matches[start + 0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start + 0][1] == '>=':
                interval_3 = '[' + str(matches[start + 0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start + 0][1] == '<':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start + 0][2]) + ')'
            if matches[start + 0][1] == '<=':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start + 0][2]) + ']'
        else:
            if (matches[start + 0][1] == '>') & (matches[start + 1][1] == '<'):
                interval_3 = '(' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ')'
            if (matches[start + 0][1] == '>') & (matches[start + 1][1] == '<='):
                interval_3 = '(' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ']'
            if (matches[start + 0][1] == '>=') & (matches[start + 1][1] == '<'):
                interval_3 = '[' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ')'
            if (matches[start + 0][1] == '>=') & (matches[start + 1][1] == '<='):
                interval_3 = '[' + str(matches[start + 0][2]) + ', ' + str(matches[start + 1][2]) + ']'

    return interval_1, interval_2, interval_3


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


# %% 2. 简化区间后的
feature_1 = 'H2S'
feature_2 = 'LOC'
feature_3 = 'TEM'
feature_1_range = H2S_range
feature_2_range = LOC_range
feature_3_range = TEM_range

rule_H2S_LOC_TEM['H2S'] = ''
rule_H2S_LOC_TEM['Location'] = ''
rule_H2S_LOC_TEM['Temperature'] = ''

for i, rule in enumerate(rule_H2S_LOC_TEM['rule']):
    matches_raw = re.findall(r'(H2S|LOC|TEM)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches = sorted(matches_raw, key=lambda x: (x[0], float(x[2])))  # sort according the letter and float number
    rule_H2S_LOC_TEM.loc[
        i, ['H2S', 'Location', 'Temperature',
                       ]] = rule_to_interval_three(matches, feature_1, feature_2,
                                                                                 feature_3, feature_1_range,
                                                                                 feature_2_range, feature_3_range)
rule_H2S_LOC_TEM.to_excel('5.1 rule_H2S_LOC_TEM（转区间后）.xlsx')
rule_H2S_LOC_TEM



# %%
# %%
print('rule_H2S_LOC_TEM的H2S独特区间：')
print(rule_H2S_LOC_TEM.sort_values('H2S')['H2S'].unique())

# %%
H2S_LOC_TEM_interval_simply_H2S = pd.DataFrame({'Raw interval': ['(10.0, 20.0]', '(10.0, 50]', '(12.5, 50]' ,'(17.5, 50]', '(2.5, 37.5]',
 '(2.5, 50]', '(20.0, 50]', '(32.5, 50]', '(37.5, 50]' ,'(7.5, 50]',
 '[0, 10.0]', '[0, 12.5]' ,'[0, 32.5]' ,'[0, 37.5]' ,'[0, 7.5]'], })

# 合并区间，阈值为这个特征最大值和最小值差值的5%，小于就会被合并
H2S_LOC_TEM_interval_simply_H2S.loc[0:, 'Simply interval'] = simple_interval_func(
    H2S_LOC_TEM_interval_simply_H2S['Raw interval'].values[0:],
    threshold=(H2S_range[1] - H2S_range[0]) * 0.05)

len(H2S_LOC_TEM_interval_simply_H2S['Simply interval'].unique())

# %%
H2S_LOC_TEM_interval_simply_H2S.to_excel('6.1 H2S_LOC_TEM_interval_simply_Have（简化区间）.xlsx')
H2S_LOC_TEM_interval_simply_H2S

# %% replace the interval for better visualization
rule_H2S_LOC_TEM_simply = rule_H2S_LOC_TEM.copy()
for i, item in enumerate(rule_H2S_LOC_TEM_simply['H2S']):
    index = list(H2S_LOC_TEM_interval_simply_H2S['Raw interval']).index(item)
    rule_H2S_LOC_TEM_simply.loc[i, 'H2S'] = H2S_LOC_TEM_interval_simply_H2S.loc[index, 'Simply interval']
rule_H2S_LOC_TEM_simply


# %%% 简化区间2 pH
print('rule_H2S_LOC_TEM的LOC独特区间：')
print(rule_H2S_LOC_TEM.sort_values('Location')['Location'].unique())

H2S_LOC_TEM_interval_simply_LOC = pd.DataFrame({'Raw interval': ['(0.5, 1]', '[0, 0.5]'], })

# 合并区间，阈值为这个特征最大值和最小值差值的5%，小于就会被合并
H2S_LOC_TEM_interval_simply_LOC.loc[0:, 'Simply interval'] = simple_interval_func(
    H2S_LOC_TEM_interval_simply_LOC['Raw interval'].values[0:],
    threshold=(LOC_range[1] - LOC_range[0]) * 0.05)

len(H2S_LOC_TEM_interval_simply_LOC['Simply interval'].unique())

# %%
H2S_LOC_TEM_interval_simply_LOC.to_excel('6.2 H2S_LOC_TEM_interval_simply_LOC（简化区间）.xlsx')
H2S_LOC_TEM_interval_simply_LOC

# %% replace the interval for better visualization
rule_H2S_LOC_TEM_simply = rule_H2S_LOC_TEM.copy()
for i, item in enumerate(rule_H2S_LOC_TEM_simply['Location']):
    index = list(H2S_LOC_TEM_interval_simply_LOC['Raw interval']).index(item)
    rule_H2S_LOC_TEM_simply.loc[i, 'Location'] = H2S_LOC_TEM_interval_simply_LOC.loc[index, 'Simply interval']
rule_H2S_LOC_TEM_simply



# %% 简化区间3 Time
# %%
print('rule_H2S_LOC_TEM的TEM独特区间：')
print(rule_H2S_LOC_TEM.sort_values('Temperature')['Temperature'].unique())

# %%
H2S_LOC_TEM_interval_simply_TEM = pd.DataFrame({'Raw interval': ['(21.0, 27.5]', '(21.0, 30]', '(23.5, 30]', '(27.5, 30]', '[17, 27.5]'], })

# 合并区间，阈值为这个特征最大值和最小值差值的5%，小于就会被合并
H2S_LOC_TEM_interval_simply_TEM.loc[0:, 'Simply interval'] = simple_interval_func(
    H2S_LOC_TEM_interval_simply_TEM['Raw interval'].values[0:],
    threshold=(TEM_range[1] - TEM_range[0]) * 0.05)

len(H2S_LOC_TEM_interval_simply_TEM['Simply interval'].unique())

# %%
H2S_LOC_TEM_interval_simply_TEM.to_excel('6.3 H2S_LOC_TEM_interval_simply_TEM(简化区间).xlsx')
H2S_LOC_TEM_interval_simply_TEM

# %% replace the interval for better visualization

for i, item in enumerate(rule_H2S_LOC_TEM_simply['Temperature']):
    index = list(H2S_LOC_TEM_interval_simply_TEM['Raw interval']).index(item)
    rule_H2S_LOC_TEM_simply.loc[i, 'Temperature'] = H2S_LOC_TEM_interval_simply_TEM.loc[index, 'Simply interval']
rule_H2S_LOC_TEM_simply


# %%
rule_H2S_LOC_TEM_combnination = rule_H2S_LOC_TEM_simply.copy()

for i in range(rule_H2S_LOC_TEM_combnination.shape[0]):
    rule_H2S_LOC_TEM_combnination.loc[i, 'Combination'] = rule_H2S_LOC_TEM_combnination.loc[
                                                        i, 'H2S'] + ' and ' + \
                                                    rule_H2S_LOC_TEM_combnination.loc[i, 'Temperature']

rule_H2S_LOC_TEM_combnination

# %%
rule_map_H2S_LOC_TEM_importance = pd.DataFrame(0, index=list(rule_H2S_LOC_TEM_combnination['Combination'].unique()),
                                         columns=list(H2S_LOC_TEM_interval_simply_LOC['Simply interval'].unique()), )
rule_map_H2S_LOC_TEM_frequency = pd.DataFrame(0, index=list(rule_H2S_LOC_TEM_combnination['Combination'].unique()),
                                        columns=list(H2S_LOC_TEM_interval_simply_LOC['Simply interval'].unique()), )
rule_map_H2S_LOC_TEM_coefficient = pd.DataFrame(0, index=list(rule_H2S_LOC_TEM_combnination['Combination'].unique()),
                                          columns=list(H2S_LOC_TEM_interval_simply_LOC['Simply interval'].unique()), )
rule_map_H2S_LOC_TEM_importance

print('检查rule_map_H2S_LOC_TEM_importance')
print(rule_map_H2S_LOC_TEM_importance.columns)


# %%
for i in range(rule_H2S_LOC_TEM_combnination.shape[0]):
    rule_map_H2S_LOC_TEM_importance.loc[
        rule_H2S_LOC_TEM_combnination.loc[i, 'Combination'], rule_H2S_LOC_TEM_combnination.loc[i, 'Location']] += \
    rule_H2S_LOC_TEM_combnination.loc[i, 'importance'] / 10
    rule_map_H2S_LOC_TEM_coefficient.loc[
        rule_H2S_LOC_TEM_combnination.loc[i, 'Combination'], rule_H2S_LOC_TEM_combnination.loc[i, 'Location']] += \
    rule_H2S_LOC_TEM_combnination.loc[i, 'coef'] / 10
    rule_map_H2S_LOC_TEM_frequency.loc[
        rule_H2S_LOC_TEM_combnination.loc[i, 'Combination'], rule_H2S_LOC_TEM_combnination.loc[i, 'Location']] += 1 / 10

rule_map_H2S_LOC_TEM_importance



# %%
rule_map_H2S_LOC_TEM_plot = pd.DataFrame(
    columns=['Location', 'Combination', 'Importance', 'Coefficient', 'Frequency'])
for Com_item in list(rule_H2S_LOC_TEM_combnination['Combination'].unique()):
    for LOC_item in list(H2S_LOC_TEM_interval_simply_LOC['Simply interval'].unique()):
        temp_rule_map = pd.DataFrame({'Location': LOC_item,
                                      'Combination': Com_item,
                                      'Importance': rule_map_H2S_LOC_TEM_importance.loc[Com_item, LOC_item],
                                      'Coefficient': rule_map_H2S_LOC_TEM_coefficient.loc[Com_item, LOC_item],
                                      'Frequency': rule_map_H2S_LOC_TEM_frequency.loc[Com_item, LOC_item],
                                      }, index=[0])
        rule_map_H2S_LOC_TEM_plot = pd.concat([rule_map_H2S_LOC_TEM_plot, temp_rule_map], ignore_index=True)
rule_map_H2S_LOC_TEM_plot

# %%
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 1))
palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

if max([abs(rule_map_H2S_LOC_TEM_coefficient.min().min()), abs(rule_map_H2S_LOC_TEM_coefficient.max().max())]) > larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_H2S_LOC_TEM_coefficient.min().min()), abs(rule_map_H2S_LOC_TEM_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_H2S_LOC_TEM_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_H2S_LOC_TEM_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_H2S_LOC_TEM_plot, x="Combination", y="Location", hue="Coefficient", linewidth=0.2,
    size="Importance", sizes=(20, 250 * rule_map_H2S_LOC_TEM_importance.max().max() / larger_import_limit),
    ax=ax, palette=palette, hue_norm=(-larger_coeff_limit, larger_coeff_limit),
)

plt.xticks(rotation=45, ha='right')

# 设置x轴和y轴标签
plt.xlabel("H2S (ppm) and Temperature (ºC)", fontsize=12)
plt.ylabel("Location", fontsize=12)

plt.xlim(-0.5, 21.5)
plt.ylim(-0.5, 1.5)


plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left',
           borderaxespad=0, labelspacing=0.5, scatterpoints=1,
           frameon=False, ncol=1, handletextpad=0.2
           )


norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm, )
sm.set_array([])
# 修改颜色条位置
fig.colorbar(sm, ax=ax, label='Coefficient', shrink=0.7, location='right', pad=0.02, anchor=(0, 0.15))


ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_H2S_LOC_TEM（简化区间）.jpg", dpi=600, bbox_inches='tight')