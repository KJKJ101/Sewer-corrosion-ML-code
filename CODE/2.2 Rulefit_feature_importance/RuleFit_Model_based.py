# %%处理 rulefit 生成的规则，计算特征重要性
# 提醒，由于画图时才发现，特征名字需要调整统一格式，但是之前rulefit建模的时候没有调，所以此处用的还是旧版的特征名字格式，为了画图，原数据集已经改了新版，但是rulefit建模得到的rule中还是旧版，所以提醒一下
import pandas as pd
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from sklearn import metrics
from imodels import RuleFitClassifier

import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#设置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 处理另存所有的规则文件
# 找出文件中所有的 rule 文件
file_list = os.listdir('./Model_RuleFit')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1, 11, 1):
    exec("temp_pd = pd.read_excel(\"./Model_RuleFit/rules_{}.xlsx\",index_col = 0,)".format(i))
    # 添加列，用于标识规则是属于哪个模型
    temp_pd['Model'] = i
    df_list.append(temp_pd)

# 将所有的规则文件合并在一起
rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ", str(rules_all.shape[0]))

# %% 计算每个模型生成的规则数

model_rule_number = pd.DataFrame(index=range(1, 11), columns=['Rule number'])
for i in range(1, 11):
    model_rule_number.loc[i, 'Rule number'] = rules_all[rules_all['Model'] == i].shape[0]
model_rule_number

# %% 保存规则数量
model_rule_number.to_excel('1. 规则数量Rule_number_速率.xlsx')

# %%
# 计算规则中包含的特征数量的函数
def find_mk(input_vars: list, rule: str):
    var_count = 0
    for var in input_vars:
        if var in rule:
            var_count += 1
    return (var_count)

# 计算特征重要性的函数
def get_feature_importance(feature_set: list, rule_set: pd.DataFrame, scaled=False):
    feature_imp = list()
    # 计算每条规则中包含的特征数量
    rule_feature_count = rule_set.rule.apply(lambda x: find_mk(feature_set, x))
    # 遍历所有的特征（feature_set），计算每个特征的重要性。
    for feature in feature_set:

        # find subset of rules that apply to a feature
        # 找到所有包含当前特征的规则
        feature_rk = rule_set.rule.apply(lambda x: feature in x)

        # find importance of linear features
        # 获取所有线性特征（类型为 'linear'）的权重
        linear_imp = rule_set[(rule_set.type == 'linear') & (rule_set.rule == feature)].importance.values

        # find the importance of rules that contain feature
        # 获取所有规则特征的权重
        rule_imp = rule_set.importance[(rule_set.type == 'rule') & feature_rk]

        # find the number of features in each rule that contain feature
        # 获取包含当前特征的规则中，特征数量的计数值（每条规则中包含的特征数）
        m_k = rule_feature_count[(rule_set.type == 'rule') & feature_rk]
        print("linear_imp.shape")
        print(linear_imp.shape)
        print("(rule_imp / m_k).shape")
        print((rule_imp / m_k).shape)

        # sum the linear and rule importances, divided by m_k
        if len(linear_imp) == 0:
            linear_imp = 0
        else:
            linear_imp = linear_imp.sum()
        # sum the linear and rule importances, divided by m_k
        if len(rule_imp) == 0:
            feature_imp.append(float(linear_imp))
        else:
            rule_imp_sum = (rule_imp / m_k).sum()  # 求和，确保结果是标量
            feature_imp.append(float(linear_imp + rule_imp_sum))  # 相加，确保结果是标量

    if scaled:
        feature_imp = 100 * (feature_imp / np.array(feature_imp).max())

    return (feature_imp)


# %%
# 读取文件
file_path = r"F:\ML2024.01\test\dataset\2024.11.23-小数据模型\总体\腐蚀速率\r.xlsx"
data = pd.read_excel(file_path, index_col=0)
data

# 去除不显著特征
X = data.drop(columns=['CorrRate'])
print(len(X.columns))
print(X.columns)

# %%
# 特征名称
feature_names = X.columns
# 计算特征重要性：用get_feature_importance() 函数
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index=feature_names, columns=['importance']).sort_values(
    by='importance', ascending=False)
# 除以10? 进行缩放？
importance_df['importance'] = importance_df['importance'] / 10
importance_df

# %% 开始画图，画特征重要性的图
# 设置颜色
palette = sns.diverging_palette(220, 20, n=7, )
hex_values = sns.color_palette(palette, as_cmap=True).as_hex()
# Print the HEX values
for hex_code in hex_values:
    print(hex_code)

# %% 定义一个颜色方案，用于后续的柱状图。
color_list = ['#c3553a', '#d28976', '#e3beb4', '#b5ccd3', '#79a5b3', '#3f7f93', ]
# 图的尺寸
fig, ax1 = plt.subplots(figsize=(3.2, 1.5))

# 画图，保存
plt.bar(importance_df.index, importance_df['importance'], width=0.4, color=color_list, edgecolor='k', alpha=0.6)
plt.xticks(rotation=45, ha='right')
#
# plt.tight_layout()

plt.ylim(0, importance_df['importance'].max() * 1.15)
plt.ylabel('Feature importance', fontsize=10)
plt.title("r", loc='center', fontsize=10)

fig.savefig("./Image/RuleFit_feature_importance.jpg", dpi=600, bbox_inches='tight')

# %% 这个也保存了，特征重要性
importance_df['Model'] = '速率Rulefit'
importance_df.to_excel('2. 特征重要性_feature_importance速率.xlsx')
importance_df

# ##替换前查看唯一值
print(rules_all['rule'].unique())

# %% simplify the rules
# 简化 rules，用首字母代替 特征名字,
# 提醒，由于画图时才发现，特征名字需要调整统一格式，但是之前rulefit建模的时候没有调，所以此处用的还是旧版的特征名字格式，为了画图，原数据集已经改了新版，但是rulefit建模得到的rule中还是旧版，所以提醒一下

rules_all['rule'] = rules_all['rule'].str.replace('Location', 'LOC', regex=False)


rules_all['rule'] = rules_all['rule'].str.replace('H2S', 'H2S', regex=False)


rules_all['rule'] = rules_all['rule'].str.replace('RH', 'RH', regex=False)

rules_all['rule'] = rules_all['rule'].str.replace('Temperature', 'TEM', regex=False)


rules_all


# 保存替换后的数据
rules_all.to_excel('./3. 简写替换processed_rules_all速率.xlsx')

# ##替换后再次查看唯一值
print(rules_all['rule'].unique())

# ##检查是否还包含原始字符串
original_strings = [ 'Location',
       'H2S',  'RH',
       'Temperature' ]
for string in original_strings:
    if rules_all['rule'].str.contains(string).any():
        print(f"Warning: '{string}' still exists")

# 计算规则重要性，图6a2 小图 就是下面的柱状图
# %% 计算规则中包含了多少种特定的字母（特征符号）

# def count_feature_letters(s):
#     letters = ['Hmax', 'Have', 'Hmin', 'Tmax', 'Tave', 'Tmin','Rmax', 'Rave', 'Rmin','pH', 'T']
#     # 使用列表推导来检查每个特征符号，确保不会误识别为单个字母
#     return len([feature for feature in letters if feature in s])
def count_feature_letters(s):
    letters = [ 'LOC',  'H2S',  'RH','TEM']
    # 使用正则表达式确保更精确的匹配
    import re
    return len(set(re.findall(r'\b(' + '|'.join(map(re.escape, letters)) + r')\b', s)))

# %% 为每条规则计算特征数量并将其存储在 rules_all DataFrame 的新列 Feature number 中。
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all

# %% 检查规则中涉及的特征数量的不同种类，看看有多少种不同的特征数量。
rules_all['Feature number'].unique()

# %% 计算每个特征数量对应的规则重要性
rule_number_importance = pd.DataFrame(columns=['Rule number', 'Importance'], index=range(6))
rule_number_importance.loc[0, :] = [1, rules_all[rules_all['Feature number'] == 1]['importance'].sum() / 10]
rule_number_importance.loc[1, :] = [2, rules_all[rules_all['Feature number'] == 2]['importance'].sum() / 10]
rule_number_importance.loc[2, :] = [3, rules_all[rules_all['Feature number'] == 3]['importance'].sum() / 10]
rule_number_importance.loc[3, :] = [4, rules_all[rules_all['Feature number'] == 4]['importance'].sum() / 10]
rule_number_importance.loc[4, :] = [5, rules_all[rules_all['Feature number'] == 5]['importance'].sum() / 10]
rule_number_importance.loc[5, :] = [6, rules_all[rules_all['Feature number'] == 6]['importance'].sum() / 10]
rule_number_importance

# %% 画柱状图
fig, ax1 = plt.subplots(figsize=(2, 1.2))
plt.bar(rule_number_importance['Rule number'], rule_number_importance['Importance'],
        width=0.4, color=color_list, edgecolor='k', alpha=0.6)

plt.xlabel('Feature number')
plt.xticks(range(1, len(rule_number_importance) + 1))
plt.ylim(0, rule_number_importance['Importance'].max() * 1.1)
plt.ylabel('Rule importance', fontsize=9)
plt.title("r", loc='center', fontsize=9)

fig.savefig("./Image/RuleFit_number_importance_.jpg", dpi=600, bbox_inches='tight')

# %% 保存特征数量的重要性
rule_number_importance['Model'] = '速率Rulefit'
rule_number_importance.to_excel('4. 数量重要性_number_importance速率.xlsx')
rule_number_importance

# %% 筛选涉及 1 个特征的规则：
rules_all_one = rules_all[rules_all['Feature number'] == 1]
rules_all_one

# %% 筛选涉及 2 个特征的规则：
rules_all_two = rules_all[rules_all['Feature number'] == 2]
rules_all_two

# %% 筛选涉及 3 个特征的规则：
rules_all_three = rules_all[rules_all['Feature number'] == 3]
rules_all_three

# %% 建立字典，将每个特征的字母符号与其描述对应起来，方便后续查看和处理。
feature_dict = {
                'LOC': 'Location',

                'H2S': 'H2S',

                'RH': 'RH',

                'TEM': 'Temperature',
                }
feature_dict


# 1个
# %% rules containing one feature
# 创建包含 1 个特征的规则的重要性表格：
col1 = feature_dict.keys()
rules_one_importance = pd.DataFrame({'Feature 1': col1, })
rules_one_importance

print("Rules with Feature Names:")
print(rules_one_importance)

# %% 计算每个特征在涉及 1 个特征的规则中的总重要性
for i in range(0, rules_one_importance.shape[0]):
    index = []
    for j in range(0, rules_all_one.shape[0]):
        if (rules_one_importance.loc[i, 'Feature 1'] in rules_all_one.iloc[j, 0]):
            index.append(j)
    rules_one_importance.loc[i, 'Importance'] = rules_all_one.iloc[index, :]['importance'].sum() / 10
rules_one_importance

# %% 输出所有特征的总重要性：所有涉及 1 个特征的规则的重要性总和
rules_one_importance['Importance'].sum()

# 输出每个特征的重要性
print("Feature Importance (for each feature):")
print(rules_one_importance)

# 输出所有特征的总重要性
total_importance = rules_one_importance['Importance'].sum()
print("Total Importance of all features:")
print(total_importance)

# %% rules containing two features
# 按重要性排序并输出：对特征进行排序，输出按重要性排序后的结果。
rules_one_importance = rules_one_importance.sort_values(by='Importance', ascending=False)
rules_one_importance

# 2个
# %% 生成包含 2 个特征的规则组合：
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 2))
col1, col2 = zip(*combinations)
rules_two_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2})
rules_two_importance

# %% 计算每个特征在涉及 2 个特征的规则中的总重要性
for i in range(0, rules_two_importance.shape[0]):
    index = []
    for j in range(0, rules_all_two.shape[0]):
        if (rules_two_importance.loc[i, 'Feature 1'] in rules_all_two.iloc[j, 0]) and (
                rules_two_importance.loc[i, 'Feature 2'] in rules_all_two.iloc[j, 0]):
            index.append(j)
    rules_two_importance.loc[i, 'Importance'] = rules_all_two.iloc[index, :]['importance'].sum() / 10
rules_two_importance

# %%
rules_two_importance['Importance'].sum()

# %%
rules_two_importance = rules_two_importance.sort_values(by='Importance', ascending=False)
rules_two_importance

# 3个
# %% 生成包含 3 个特征的规则组合：
# %% rules containing three features
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 3))
col1, col2, col3 = zip(*combinations)
rules_three_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2, 'Feature 3': col3})
rules_three_importance

# %%
for i in range(0, rules_three_importance.shape[0]):
    index = []
    for j in range(0, rules_all_three.shape[0]):
        if (rules_three_importance.loc[i, 'Feature 1'] in rules_all_three.iloc[j, 0]) and (
                rules_three_importance.loc[i, 'Feature 2'] in rules_all_three.iloc[j, 0]) and (
                rules_three_importance.loc[i, 'Feature 3'] in rules_all_three.iloc[j, 0]):
            index.append(j)
    rules_three_importance.loc[i, 'Importance'] = rules_all_three.iloc[index, :]['importance'].sum() / 10
rules_three_importance

# %%
rules_three_importance = rules_three_importance.sort_values(by='Importance', ascending=False)
rules_three_importance

# 绘图，将上述的计算绘柱状图：涉及不同数量特征的规则组合及其重要性
# %% 创建空数据网，储存数据
feaure_combinations = pd.DataFrame(columns=['Feature (combinations)', 'Importance'], index=range(
    len(rules_one_importance) + len(rules_two_importance) + len(rules_three_importance)))
feaure_combinations

# %% 填充涉及 1 个特征的规则组合：
feaure_combinations.iloc[0:len(rules_one_importance), 0] = rules_one_importance['Feature 1']
feaure_combinations.iloc[0:len(rules_one_importance), 1] = rules_one_importance['Importance']

# %% 填充涉及 2 个特征的规则组合：
feaure_combinations.iloc[len(rules_one_importance):len(rules_one_importance) + len(rules_two_importance),
0] = rules_two_importance.apply(lambda row: row['Feature 1'] + ' and ' + row['Feature 2'], axis=1)
feaure_combinations.iloc[len(rules_one_importance):len(rules_one_importance) + len(rules_two_importance), 1] = \
rules_two_importance['Importance']

# %% 填充涉及 3 个特征的规则组合：
feaure_combinations.iloc[len(rules_one_importance) + len(rules_two_importance):, 0] = rules_three_importance.apply(
    lambda row: row['Feature 1'] + ', ' + row['Feature 2'] + ' and ' + row['Feature 3'], axis=1)
feaure_combinations.iloc[len(rules_one_importance) + len(rules_two_importance):, 1] = rules_three_importance[
    'Importance']

# 确保 Importance 列的类型为 float，方便后续处理和绘图。
feaure_combinations['Importance'] = feaure_combinations['Importance'].astype('float')
feaure_combinations
# 保存到 Excel 文件
feaure_combinations.to_excel('5. 特征联合重要性feaure_combinations速率.xlsx')

# %% 筛选出重要性大于 0.05 的规则组合：
feaure_combinations_plot = feaure_combinations[feaure_combinations['Importance'] > 0]
feaure_combinations_plot.to_excel('6. 特征联合重要性feaure_combination速率（大于0筛选后）.xlsx')

# 定义用于根据重要性值设置颜色的函数
# 为每个规则组合的重要性值分配一个颜色，这样绘图时就能直观地显示规则的重要性。
def colors_from_values(values, palette_name):
    normalized = (values - min(values)) / (max(values) - min(values))
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# 绘柱状图： 图6 a3
fig, ax = plt.subplots(figsize=(1.6, 4.7))

s = sns.barplot(x=feaure_combinations_plot['Importance'],
                y=feaure_combinations_plot['Feature (combinations)'],
                palette=colors_from_values(feaure_combinations_plot['Importance'], "vlag"),
                orient='h', alpha=0.6, edgecolor=".1", width=0.6)
s.set_yticklabels(s.get_yticklabels(), size=9)
plt.xlim(0, feaure_combinations_plot['Importance'].max() * 1.1)

plt.xlabel('')
plt.xlabel("Rule importance", fontsize=9)
plt.title("r", fontsize=9, loc='center')

plt.margins(0.02)
fig.savefig("./Image/RuleFit_detailed_importance.jpg", dpi=600, bbox_inches='tight')

# %%