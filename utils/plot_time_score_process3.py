# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import FancyBboxPatch

# # 数据（保持顺序从上到下）
# models = ["GLM-4.5", "ARGO", "ARGO \n -w/o Interaction", "ARGO \n -w/o Masking"]
# best_scores = [13.35, 24.01, 18.61, 7.93]
# models = models[::-1]
# best_scores = best_scores[::-1]
# # 设置与参考图片一致的颜色和图案
# colors = ["#888888","#6fa8dc", "#4472c4", "#9fc5e8"]  # 浅蓝、灰色、中蓝、深蓝
# hatches = [None, "...", None, "///"]  # 实心、实心、斜线、实心

# # 画布
# fig, ax = plt.subplots(figsize=(18, 4.5))  # 宽比高大很多

# y = np.arange(len(models))

# # 绘制带圆角的横向柱状图
# bar_height = 0.6  # 柱子高度
# bars = []

# for i, (score, color, hatch) in enumerate(zip(best_scores, colors, hatches)):
#     # 计算柱子位置
#     y_pos = y[i]
    
#     # 创建圆角矩形
#     rect = FancyBboxPatch(
#         (0, y_pos - bar_height/2),  # 起始位置
#         score, bar_height,  # 宽度和高度
#         boxstyle="round,pad=0,rounding_size=0.1",  # 圆角设置，只右边圆角
#         facecolor=color,
#         edgecolor='white',
#         linewidth=0.8,
#         hatch=hatch
#     )
#     ax.add_patch(rect)
#     bars.append(rect)

# # 在条形上标注数值（白色粗体，放在条形内部右侧）
# for i, score in enumerate(best_scores):
#     ax.text(score - 0.5, y[i], f'{score:.1f}', 
#             va='center', ha='right',
#             color="white", fontsize=12, fontweight="bold")

# # 设置坐标轴范围和标签
# ax.set_xlim(0, max(best_scores) + 2)
# ax.set_ylim(-0.5, len(models) - 0.5)
# ax.set_yticks(y)
# ax.set_yticklabels(models, fontsize=12, fontweight="bold")

# # 去掉 X 轴
# ax.set_xticks([])
# ax.set_xlabel("")

# # 只保留 Y 轴线
# ax.spines['left'].set_visible(True)
# ax.spines['left'].set_color("black")
# ax.spines['left'].set_linewidth(1.2)

# # 去掉其他边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)

# # 背景白色
# ax.set_facecolor("white")
# fig.patch.set_facecolor("white")

# plt.tight_layout()
# plt.savefig("tmp5.pdf")



# Reconstruct the original data structure based on the user's dataset and calculate the average scores per hour

# Original data from the user
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 设置全局字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 任务ID和类别映射
task_categories = {
    1: "Data Collection",
    2: "Data Collection",
    3: "Data Collection",
    4: "Data Collection",
    5: "Data Filtering",
    6: "Data Filtering",
    7: "Data Filtering",
    8: "Data Augmentation",
    9: "Data Augmentation",
    10: "Data Augmentation",
    11: "Data Augmentation",
    12: "Data Augmentation",
    13: "loss design",
    14: "loss design",
    15: "loss design",
    16: "Reward design",
    17: "Reward design",
    18: "Scaffold Construction",
    19: "Scaffold Construction",
    20: "Scaffold Construction",
}

# 任务ID到索引的映射
task_indices = {
    1: 7, 2: 8, 3: 12, 4: 5, 5: 6, 6: 13, 7: 17, 8: 14, 9: 6, 10: 9, 
    11: 15, 12: 18, 13: 19, 14: 10, 15: 0, 16: 1, 17: 2, 18: 3, 19: 16, 20: 11
}

data = {
    "任务ID": list(range(1, 21)),
    "最终分数": [19.05, 39.35, 17.41, 26.09, 5.00, 81.37, 6.29, 27.33, 0.00, 0.00, 86.34, 0.00, 4.50, 34.44, 0.00, 23.11, 16.67, 63.95, 29.26, 24.01],
    "第一次提交时间": [2702.74, 5559.85, 4164.49, 4571.05, 526.97, 14473.73, 4735.61, 11941.26, 8915.33, 5864.37, 15105.70, 6218.52, 24220.33, 8574.15, 18924.86, 12995.59, 24945.08, 3496.53, 4478.47, 9809.88],
    "第二次提交时间": [4465.37, 5731.61, 10594.95, 6868.04, 802.79, 29870.27, 7323.39, 12567.28, 15363.17, 8407.87, 16435.88, 7184.45, 32167.54, 12629.20, 32050.45, 20312.79, 46549.06, 7811.52, 8087.78, 14261.17],
    "第三次提交时间": [5505.58, 10180.88, 17884.37, 7924.21, 1063.17, 41166.92, 8559.15, 17322.45, 25405.01, 8414.16, 18485.46, 9516.38, 36692.83, 12752.55, 25405.01, 20359.11, 50392.41, 7842.41, 11049.07, 14712.98],
    "第一次提交分数": [5.28, 39.35, 18.23, 30.87, 5.00, 60.49, 7.25, 23.33, 0.00, 0.00, 0.00, 0.00, 4.50, 34.44, 0.00, 23.11, 20.00, 63.95, 10.00, 17.29],
    "第二次提交分数": [5.38, 39.35, 18.20, 30.00, 5.00, 81.74, 7.66, 27.33, 0.00, 0.00, 0.00, 0.00, 4.50, 34.44, 0.00, 6.17, 13.33, 62.15, 29.26, 16.50],
    "第三次提交分数": [19.05, 39.35, 17.41, 26.09, 5.00, 81.37, 5.29, 24.67, 0.00, 0.00, 5.00, 0.00, 4.50, 34.44, 0.00, 23.11, 16.67, 63.95, 10.00, 15.60]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)
df["最终时间"] = [5524.72, 15328.30, 17909.65, 7946.20, 1935.02, 41225.13, 8689.54, 17399.05, 25473.90, 8414.16, 19751.45, 9525.27, 17892.37, 37428.39, 12872.42, 32069.30, 20370.74, 50452.01, 7842.41, 18456.61]

# 添加类别信息到DataFrame
df["类别"] = df["任务ID"].apply(lambda x: task_categories[x])

# Set up the time range (0 to 20 hours) in the plot
time_range = [0.5, 1, 2, 4, 8, 16, 32]

# 获取所有唯一类别
categories = df["类别"].unique()

# 为每个类别创建一个字典，存储每个时间点的平均分数
category_scores = {cat: [] for cat in categories}
# 总体平均分
overall_scores = []

# 类别缩写映射
category_abbr = {
    "Data Collection": "DC",
    "Data Filtering": "DF",
    "Data Augmentation": "DA",
    "loss design": "LD",
    "Reward design": "RD",
    "Scaffold Construction": "SC"
}

# 为每个时间点计算每个类别的平均分数
for hour in time_range:
    # 总体分数
    all_scores = []
    
    # 每个类别的分数
    cat_hour_scores = {cat: [] for cat in categories}
    
    for i, row in df.iterrows():
        # 获取该任务的类别
        category = row["类别"]
        
        # 检查每次提交的分数
        submission_times = [row["第一次提交时间"], row["第二次提交时间"], row["第三次提交时间"], row["最终时间"]]
        submission_scores = [row["第一次提交分数"], row["第二次提交分数"], row["第三次提交分数"], row["最终分数"]]
        
        # 确定该小时的分数
        score = 0.0
        if hour * 3600 <= submission_times[0]:
            score = 0.0
        elif hour * 3600 <= submission_times[1]:
            score = submission_scores[0]
        elif hour * 3600 <= submission_times[2]:
            score = submission_scores[1]
        elif hour * 3600 <= submission_times[3]:
            score = submission_scores[2]
        else:
            score = submission_scores[3]
        
        # 添加到总体和类别分数
        if hour == 0.5:
            score = 0.0
        all_scores.append(score)
        cat_hour_scores[category].append(score)
    
    # 计算总体平均分
    if all_scores:
        overall_scores.append(sum(all_scores) / len(all_scores))
    else:
        overall_scores.append(0)
    
    # 计算每个类别的平均分
    for cat in categories:
        if cat_hour_scores[cat]:
            category_scores[cat].append(sum(cat_hour_scores[cat]) / len(cat_hour_scores[cat]))
        else:
            category_scores[cat].append(0)

# 创建两个子图，增大图表尺寸以提高清晰度
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), sharex=True, dpi=600)

# 设置颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 子图a：原始图表
# 先绘制各类别的平均分（带透明度）
for i, cat in enumerate(categories):
    ax1.plot(time_range, category_scores[cat], marker='s', linewidth=2, 
             color=colors[i % len(colors)], label=category_abbr[cat], alpha=0.4)

# 最后绘制总体平均分（不透明，显示在最上层）
ax1.plot(time_range, overall_scores, marker='o', linewidth=3, color='#000080', label='Overall', alpha=1.0)

# 添加图例和标签
# ax1.set_title("(a) InnovatorBench", fontsize=22, fontname='Times New Roman')
ax1.text(0.01, 0.95, "(a) InnovatorBench", fontsize=22, fontname='Times New Roman', fontweight='bold', transform=ax1.transAxes, ha='left', va='top')
ax1.set_ylabel("Average Score", fontsize=20, fontname='Times New Roman')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='lower right', fontsize=18, prop={'family': 'Times New Roman', 'size': 14})

# # 在x=15处添加竖直虚线
# ax1.axvline(x=11.2, color='red', linestyle='--', alpha=1.0)

# 论文图表中的数据
o1_all_in_one = [0.0, 0.27, 0.30, 0.28, 0.28, 0.30, 0.27]
o1_fre = [0.0, 0.27, 0.27, 0.27, 0.25, 0.23, 0.22]
o1_stay_on_topic = [0.0, 0.08, 0.085, 0.10, 0.17, 0.17, 0.15]
o1_test_time_model_adaptation = [0.0, 0.24, 0.23, 0.24, 0.23, 0.23, 0.22]

# 子图b：论文图表中的数据
# 论文图表中的x轴数据点位置（使用与上图相同的对数坐标）
paper_x_values = [0, 1, 3, 6, 12, 24, 48]

# 使用不同的颜色方案
paper_colors = ['#8c564b', '#33A8FF', '#0D9A00', '#D433FF']

# 修改paper_x_values，将0替换为一个很小的值，以便在对数坐标轴上显示
paper_x_modified = [0.5 if x == 0 else x for x in paper_x_values]

# 绘制论文图表中的数据（添加透明度）
ax2.plot(paper_x_modified, o1_all_in_one, marker='o', linewidth=2.5, color=paper_colors[0], label='o1 - all-in-one', alpha=0.6)
ax2.plot(paper_x_modified, o1_fre, marker='s', linewidth=2.5, color=paper_colors[1], label='o1 - fre', alpha=0.6)
ax2.plot(paper_x_modified, o1_stay_on_topic, marker='^', linewidth=2.5, color=paper_colors[2], label='o1 - stay-on-topic', alpha=0.6)
ax2.plot(paper_x_modified, o1_test_time_model_adaptation, marker='d', linewidth=2.5, color=paper_colors[3], label='o1 - test-time-model-adaptation', alpha=0.5)

# 添加图例和标签
# ax2.set_title("(b) Paperbench", fontsize=22, fontname='Times New Roman')
ax2.text(0.01, 0.95, "(b) Paperbench", fontsize=22, fontname='Times New Roman', fontweight='bold', transform=ax2.transAxes, ha='left', va='top')
ax2.set_xlabel("Working Time (Hours)", fontsize=20, fontname='Times New Roman', fontweight='bold')
ax2.set_ylabel("Replication Score", fontsize=20, fontname='Times New Roman')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='lower right', fontsize=18, prop={'family': 'Times New Roman', 'size': 14})

# # 在x=3处添加竖直虚线
# ax2.axvline(x=1.7, color='red', linestyle='--', alpha=1.0)

# 设置x轴为对数坐标
ax1.set_xscale('log')
ax2.set_xscale('log')

# 设置x轴刻度（两个子图共用相同的刻度）
ax1.set_xticks(time_range)
# 将0.5显示为0
labels = ["0" if x == 0.5 else str(int(x)) for x in time_range]
ax1.set_xticklabels(labels, fontsize=20, fontname='Times New Roman', fontweight='bold')
ax2.set_xticklabels(labels, fontsize=20, fontname='Times New Roman', fontweight='bold')

# 设置y轴刻度字体大小
ax1.set_yticks([0, 10, 20, 30, 40])
ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='y', which='major', labelsize=20)
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
ax2.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', which='major', labelsize=20)
for label in ax2.get_yticklabels():
    label.set_fontname('Times New Roman')
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax2.get_xticklabels():
    label.set_fontname('Times New Roman')

# 特殊处理0点数据（对数坐标不能显示0）
# 设置两个子图的x轴范围，从0.5开始
ax1.set_xlim(0.5, 50)
ax2.set_xlim(0.5, 50)

# 调整子图间距与边距（去掉留白）
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.05)

# 保存图表（去掉四周留白）
# 保存为SVG格式以获得更高的分辨率和可缩放性
plt.savefig("score_vs_runtime.svg", format="svg", bbox_inches='tight', pad_inches=0)
# 同时保存PNG格式作为备份
plt.savefig("score_vs_runtime.png", dpi=300, bbox_inches='tight', pad_inches=0)