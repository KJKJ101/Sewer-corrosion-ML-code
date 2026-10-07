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
feature_dict = {'Hmax': 'H2SMax (mg/m3)',
                'Have': 'H2SAve(mg/m3)',
                'Hmin': 'H2SMin (mg/m3)',
                'Tmax': 'Temp.Max (ºC)',
                'Tave': 'Temp.Ave (ºC)',
                'Tmin': 'Temp.Min (ºC)',
                'Rmax': 'RHMax (%)',
                'Rave': 'RHAve (%)',
                'Rmin': 'RHMin (%)',
                'pH': 'Initial pH',
                'Time': 'Time (day)',}
feature_dict

# %% simplify the rules
rules_all['rule'] = rules_all['rule'].str.replace('H2SMax (mg/m3)', 'Hmax', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('H2SAve(mg/m3)', 'Have', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('H2SMin (mg/m3)', 'Hmin', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Temp.Max (ºC)', 'Tmax', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Temp.Ave (ºC)', 'Tave', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Temp.Min (ºC)', 'Tmin', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('RHMax (%)', 'Rmax', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('RHAve (%)', 'Rave', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('RHMin (%)', 'Rmin', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Initial pH', 'pH', regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Time (day)', 'Time', regex=False)
rules_all


# %%
def count_feature_letters(s):
    letters = ['Hmax', 'Have', 'Hmin', 'Tmax', 'Tave', 'Tmin','Rmax', 'Rave', 'Rmin','pH', 'Time']
    # 使用正则表达式确保更精确的匹配
    import re
    return len(set(re.findall(r'\b(' + '|'.join(map(re.escape, letters)) + r')\b', s)))


# %%
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all = rules_all[rules_all['importance'] > 0.1]
rules_all

# %%
# 保存过滤后的规则到 Excel 文件：
rules_all.to_excel('2.rules_all_0.1（重要性大于0.1过滤后的规则）.xlsx')

# %%
rules_two = rules_all[rules_all['Feature number'] == 2]
rules_two
# %%
# 保存过滤后的规则到 Excel 文件：
rules_two.to_excel('3.rules_one（含二个特征的规则）.xlsx')

# %%
Have_pH_index = []
Hmax_Time_index = []
Have_Time_index = []

for i, rule in enumerate(rules_two['rule']):
    if ('Have' in rule) and ('pH' in rule):
        Have_pH_index.append(i)
    if ('Hmax' in rule) and ('Time' in rule):
        Hmax_Time_index.append(i)
    if ('Have' in rule) and ('Time' in rule):
        Have_Time_index.append(i)

rule_Have_pH = rules_two.iloc[Have_pH_index, :]
rule_Have_pH = rule_Have_pH.reset_index(drop=True)

rule_Hmax_Time = rules_two.iloc[Hmax_Time_index, :]
rule_Hmax_Time = rule_Hmax_Time.reset_index(drop=True)

rule_Have_Time = rules_two.iloc[Have_Time_index, :]
rule_Have_Time = rule_Have_Time.reset_index(drop=True)

print(rule_Have_pH.shape[0], rule_Hmax_Time.shape[0], rule_Have_Time.shape[0])

# %%
rule_Have_pH.to_excel('4.rule_Have_pH（提取包含目标特征）.xlsx')

# %% the min and max value of each feature
Hmax_range = [4.3, 630]
Have_range = [1.5, 500]
Hmin_range = [0, 440]
Tmax_range = [17, 30]
Tave_range = [17, 30]
Tmin_range = [15, 30]
Rmax_range = [90, 100]
Rave_range = [90, 100]
Rmin_range = [83, 100]
pH_range = [7.8, 12]
Time_range = [8, 1613]

# 变量最大值？我这上哪知道去
# can use to set same scale for plot
# give these two variables maxmimum values
larger_coeff_limit = 0.8314820655244579
larger_import_limit = 0.3412127346229156


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
# 以上是通用部分，前处理
# 以上是通用部分，前处理
# 以上是通用部分，前处理