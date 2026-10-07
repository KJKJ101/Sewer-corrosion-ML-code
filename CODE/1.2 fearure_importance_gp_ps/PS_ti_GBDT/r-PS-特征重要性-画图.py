import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

# 配置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14  # 设置全局字体大小

# 示例数据
categories = ["H2S (ppm)",	"Temperature (ºC)",	"RH (%)"]

values = [1,		0.122279653,	0.04907281]
std_devs = [0,		0.04686164,	0.024659987]  # 标准差

# 使用 "plasma" 颜色映射并反转
color_map = cm.plasma_r  # 反转颜色映射
num_colors = len(categories)

# 获取颜色，并调整透明度
color_list = [color_map(i / num_colors) for i in range(num_colors)]
alpha = 0.5  # 设置透明度为0.5
lighter_color_list = [(r * 1, g * 1, b * 1, alpha) for r, g, b, _ in color_list]  # 调整颜色亮度并加上透明度

# 设置第一张图的图形大小
# fig_width = len(categories) * 0.5  # 基于类别数目调整宽度
fig_width = 6
fig_height = 5  # 固定高度
fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

# 创建柱状图并添加误差条，调整误差棒宽度
error_kw = {'elinewidth': 0.8}  # 设置误差棒的线宽
plt.bar(categories, values, yerr=std_devs, capsize=5, color=lighter_color_list, edgecolor='black', error_kw=error_kw, width=0.7)  # 设置柱子的宽度

# 设置标题和标签
plt.title('PS', fontsize= 14)
plt.ylabel('Average feature importance: r', fontsize= 14)

# 调整字体大小，自动适应图表大小
# 横坐标标签旋转45度
plt.xticks(rotation=45)  # 设置横坐标标签旋转45度，字体大小为12
plt.yticks()  # 设置纵坐标标签字体大小

# 自动调整图形布局，确保内容不被截断
plt.tight_layout()

# 显示图形
fig.savefig("./Image/液相-速率GBDT_feature_importance.jpg", dpi=600, bbox_inches='tight')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from matplotlib import cm
from matplotlib.colors import to_rgba
# 配置字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14  # 设置全局字体大小
# 统一颜色：每个特征固定一种颜色（以第二张图为准）
feature_colors = {
    "Location": '#FF6F61',
    "RH (%)": '#6B8E23',
    "Temperature (ºC)": '#1E90FF',
    "H2S (ppm)": '#FF1493',
}
bar_alpha = 0.7  # 柱子透明度，和第二张图散点的 0.7 一致

categories2 = [	"H2S (ppm)",	"RH (%)",	"Temperature (ºC)"]
rankings = np.array([
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2],
    [1, 3, 2]
])

# 平均名次（您提供的名次）
average_ranks = [1,	3,	2]

x_pos = np.arange(1, 11)
markers = [ 'p', '^', 's' ]


# 使用 "plasma" 颜色映射并反转
color_map = cm.plasma_r  # 反转颜色映射
num_colors = len(categories2)


colors = [ '#FF1493', '#6B8E23', '#1E90FF']  #



# 增大第二张图的宽度
fig_width_2 = 7.25  # 调整第二张图的宽度
fig_height = 5  # 固定高度
fig, ax2 = plt.subplots(figsize=(fig_width_2, fig_height))

# 绘制散点图
for i, category in enumerate(categories2):
    color = feature_colors[category]
    plt.scatter(x_pos, rankings[:, i], label=category, marker=markers[i], s=100, alpha=0.7, color=color)
    plt.plot(x_pos, rankings[:, i], linestyle='--', color=color, alpha=0.5)

    # 计算并显示平均名次
    avg_rank = average_ranks[i]
    plt.text(x_pos[-1] + 0.2, rankings[-1, i], f'{avg_rank:.1f}', color=color, fontsize=12, ha='left', va='center')

# 设置标题和标签
plt.title('PS', fontsize=14, family='Arial')
plt.xlabel('Different dataset split', fontsize=14, family='Arial')
plt.ylabel('Feature importance rank: r', fontsize=14, family='Arial')

plt.xticks(x_pos)
plt.yticks(np.arange(1, 5, 1))

# 调整x轴范围，以避免超出边界
plt.xlim(0.5, 10.9)

# 反转y轴，使得y轴从上到下增加
ax2.invert_yaxis()

# 增加图例并移出图框
plt.legend(loc='upper left', fontsize=12, bbox_to_anchor=(1.002, 1.0195), frameon=True, edgecolor='black', facecolor='none',
           labelspacing=0.5, handletextpad=0.5)

# 自动调整图形布局，确保内容不被截断
plt.tight_layout()

# 显示图形
fig.savefig("./Image/r-ps-液相-速率GBDT_Feature_Rankings_with_Avg_Rank.jpg", dpi=600, bbox_inches='tight')
plt.show()
