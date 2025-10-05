import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Claude Sonnet 4", "GPT-5", "GLM-4.5"]
times_hours = [14 + 0/60 + 52/3600, 16 + 13/60 + 34/3600, 8 + 27/60 + 8/3600]
costs = [65.09, 80.50, 18.78]
best_ranks = [1, 3, 2]
final_ranks = [1, 2, 3]

# Assign unique colors and shapes
colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
markers = ['o', 's', 'D']

# 计算并设置坐标轴范围与末端刻度，确保最大端刻度显示
x_max = max(times_hours)
y_max = max(costs)

def round_up(value, step):
    return np.ceil(value / step) * step

x_max_rounded = round_up(x_max * 1.0, 0.5)
y_max_rounded = round_up(y_max * 1.05, 5.0)

fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

for i in range(len(models)):
    ax.scatter(times_hours[i], costs[i],
               s=120, c=colors[i], marker=markers[i], edgecolors='black', label=models[i])
    label = f"(Best:{best_ranks[i]}, Final:{final_ranks[i]})"
    if i == 0:
        ax.text(times_hours[i] - 1.1, costs[i] - 2.7, label, fontsize=9)
    elif i == 2:
        ax.text(times_hours[i] + 0.17, costs[i] - 0.5, label, fontsize=9)
    else:
        ax.text(times_hours[i] - 0.8, costs[i] - 2.7, label, fontsize=9)

ax.set_xlabel("Average Time (Hours)")
ax.set_ylabel("Average Cost (USD)")

# 设置坐标轴范围，强制显示到末端，并显式设置刻度含最大值
print(x_max_rounded, y_max_rounded)
ax.set_xlim(1.5, x_max_rounded)
ax.set_ylim(4, y_max_rounded)
ax.set_xticks(np.arange(1.5, x_max_rounded + 1e-9, 0.5))
ax.set_yticks(np.arange(5, y_max_rounded + 1e-9, 5.0))

# 去除顶部/右侧脊与刻度
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

# 栅格与图例
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# 更高分辨率导出，减小模糊
fig.tight_layout()
plt.savefig("./time_cost_analysis.png", dpi=150, bbox_inches='tight')
plt.show()