# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fonts for Chinese labels
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Recreate the DataFrame from your table (for a self-contained cell) ---
# data = [
#     [1, "SFT", "纯文本", "Data Collection & Curation", "summary"],
#     [2, "SFT", "纯文本", "Data Collection & Curation", "translation"],
#     [3, "SFT", "纯文本", "Data Collection & Curation", "summary"],
#     [4, "SFT", "纯文本", "Data Collection & Curation", "Medical"],
#     [5, "Pre-train", "纯文本", "Data Cleaning", ""],
#     [6, "SFT", "纯文本", "Data Cleaning", "Math"],
#     [7, "Inference", "纯文本", "Data Cleaning", "Code"],
#     [8, "SFT", "纯文本", "DA", "Science"],
#     [9, "SFT", "文本+图像", "DA", "Logical Reasoning"],
#     [10, "SFT", "纯文本", "DA", "Deep Research"],
#     [11, "SFT", "纯文本", "DA", "Social"],
#     [12, "SFT", "文本+图像", "DA", "Science"],
#     [13, "RL", "纯文本", "LD", "Math"],
#     [14, "RL", "纯文本", "LD", "Math"],
#     [15, "RL", "纯文本", "LD", "Alignment"],
#     [16, "RL", "纯文本", "RD", "Deep Research"],
#     [17, "RL", "文本+图像", "RD", "GUI"],
#     [18, "Inference", "纯文本", "SC", "Deep Research"],
#     [19, "Inference", "文本+图像", "SC", "Math"],
#     [20, "Inference", "文本+图像", "SC", "image recognition"],
# ]
data = [
    [1, "SFT", "纯文本", "DF", "summary"],
    [2, "SFT", "纯文本", "DF", "translation"],
    [3, "SFT", "纯文本", "DF", "summary"],
    [4, "SFT", "纯文本", "DF", "Medical"],
    [5, "Pre-train", "纯文本", "DC", ""],
    [6, "SFT", "纯文本", "DC", "Math"],
    [7, "Inference", "纯文本", "DC", "Code"],
    [8, "SFT", "纯文本", "DA", "Science"],
    [9, "SFT", "文本+图像", "DA", "Logical Reasoning"],
    [10, "SFT", "纯文本", "DA", "Deep Research"],
    [11, "SFT", "纯文本", "DA", "Social"],
    [12, "SFT", "文本+图像", "DA", "Science"],
    [13, "RL", "纯文本", "LD", "Math"],
    [14, "RL", "纯文本", "LD", "Math"],
    [15, "RL", "纯文本", "LD", "Alignment"],
    [16, "RL", "纯文本", "RD", "Deep Research"],
    [17, "RL", "文本+图像", "RD", "GUI"],
    [18, "Inference", "纯文本", "SC", "Deep Research"],
    [19, "Inference", "文本+图像", "SC", "Math"],
    [20, "Inference", "文本+图像", "SC", "image recognition"],
]
df = pd.DataFrame(data, columns=["Task ID", "Algorithm", "Modality", "Task", "Domain"])

# Order the research goals by a logical sequence
# goals_order = [
#     "Data Collection & Curation",
#     "Data Cleaning",
#     "DA",
#     "LD",
#     "RD",
#     "SC",
# ]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
colors_dict = {
    "DC": "#1f77b4",
    "DF": "#ff7f0e",
    "DA": "#2ca02c",
    "LD": "#d62728",
    "RD": "#9467bd",
    "SC": "#e377c2"
}
goals_order = [
    "DF",
    "DC",
    "DA",
    "LD",
    "RD",
    "SC",
]
df["Task"] = pd.Categorical(df["Task"], categories=goals_order, ordered=True)

counts = df["Task"].value_counts().sort_values(ascending=False)
colors = [colors_dict[goal] for goal in counts.index]
# --- Plot: y轴为题目数量，x轴为研究目标 ---
fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
# bars = ax.bar(counts.index, counts.values, color="#81D8D0")
bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.4)

# Y ticks at 0,1,2,3,...
ymax = int(counts.max())
ax.set_yticks(np.arange(0, ymax + 2, 1))
ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_axisbelow(True)

# Labels and title
ax.set_ylabel("Task Count")
# ax.set_xlabel("研究目标")
# ax.set_title("柱状图：各研究目标的题目数量")

# Annotate counts on top of each bar
for rect in bars:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2.0, height + 0.05, f"{int(height)}",
            ha="center", va="bottom", fontsize=10)

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

# Save for download
out_path = "./benchmark_bar_goals.png"
fig.savefig("./benchmark_bar_goals.pdf", format="pdf")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
out_path
