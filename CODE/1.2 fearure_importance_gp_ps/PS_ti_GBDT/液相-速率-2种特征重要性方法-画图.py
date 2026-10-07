import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14  # 设置全局字体大小

# 示例数据
categories = [ 'RH (%)', 'Temperature (ºC)',  'H2S (ppm)',]

# 每个类别下的三组数据

values2 = [	-0.030445702,	0.047447576, 	0.77915988,]  # PFI
values3 = [	0.037670701,	0.091158338, 	0.526765471,]  # SHAP

# 定义每种类型的柱子颜色
colors = [ '#ff7f0e', '#1f77b4']  # RMSE, MAE, R² 的颜色

# 动态计算图的大小：基于类别数目或其他因素
fig_width = 5  # 调整宽度，根据类别数来设置
fig_height = 5.5  # 可以固定高度
fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# 每个类别下的3根柱子并排显示
width = 0.3  # 每根柱子的宽度
x_pos = np.arange(len(categories))  # x坐标位置
offsets = [-width, 0, width]  # 每组柱子的偏移量

# 创建柱状图并添加误差条，调整误差棒宽度
error_kw = {'elinewidth': 0.8}  # 设置误差棒的线宽

# 绘制每一组柱子
bars = []  # 用于存储柱子句柄以便图例
for i, (values, offset, color, label) in enumerate(zip([values2, values3], offsets, colors, [ 'PFI', 'SHAP'])):
    bar = plt.bar(x_pos + offset, values, capsize=5, color=color, edgecolor='black', error_kw=error_kw, width=width, label=label)
    bars.append(bar)  # 存储柱子句柄


# 设置标题和标签

plt.title('PS', fontsize= 14)
plt.ylabel('Relative Feature Importance Value: r', fontsize= 14)

# 设置横坐标标签
plt.xticks(x_pos, categories, rotation=45, ha='right')

# 设置纵坐标
plt.yticks()  # 设置纵坐标字体大小

# 设置纵坐标范围（根据数据调整）
plt.ylim(0, 0.9)
plt.yticks(np.arange(0, 0.91, 0.2))

# 设置横纵坐标轴的线宽
ax = plt.gca()  # 获取当前坐标轴
ax.spines['top'].set_linewidth(1)   # 设置上边界线宽
ax.spines['right'].set_linewidth(1) # 设置右边界线宽
ax.spines['left'].set_linewidth(1)  # 设置左边界线宽
ax.spines['bottom'].set_linewidth(1) # 设置下边界线宽

# 设置刻度线
ax1.tick_params(axis='both', which='both', direction='out', length=6)

# 添加图例，并将其放置在左上角，且设置外框颜色为黑色
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 1.01), frameon=True, edgecolor='black',
           labelspacing=0.1, handletextpad=0.6)

# 自动调整图形布局，确保内容不被截断
plt.tight_layout()

# 显示图形
fig.savefig("./液相-2种特征重要性方法.jpg", dpi=600, bbox_inches='tight')
plt.show()
