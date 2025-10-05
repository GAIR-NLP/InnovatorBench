# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Construct the data
rows = [
    ["CC", 23.43, 23.23, 25.48, 25.47, 4249.53, 6914.99, 10373.76, 11677.22],
    ["DC",               24.25, 31.47, 30.55, 30.89, 6578.77, 12665.48, 16929.75, 17283.23],
    ["DA",            4.67,  5.47,  7.42, 22.73, 9609.04, 11991.73, 17682.33, 16112.77],
    ["LD",                 12.98,  2.25, 19.47, 12.98, 15525.79, 22398.37, 24722.69, 22731.06],
    ["RD",                0.00,  0.00,  0.00, 10.67, 18149.06, 24741.14, 17485.46, 33215.99],
    ["SC",             31.32, 34.91, 13.34, 36.63, 10973.36, 20816.12, 30720.74, 23125.53],
    ["WA",                   16.13, 16.31, 16.45, 23.92, 10028.76, 15237.04, 19099.86, 19156.21],
]

columns = ["Research Domain", "Score 1", "Score 2", "Score 3", "Final Score",
           "Runtime 1", "Runtime 2", "Runtime 3", "Final Runtime"]

df = pd.DataFrame(rows, columns=columns)

# Separate out the weighted row and category rows
weighted_row = df[df["Research Domain"] == "WA"].copy()
cat_df = df[df["Research Domain"] != "WA"].copy()

# Prepare x-axis
x_labels = ["First", "Second", "Third", "Final"]
x = np.arange(len(x_labels))

# ---------- Chart 1: Score trends (multi-line, one plot) ----------
plt.figure(figsize=(4.8, 4))
for _, r in cat_df.iterrows():
    y = [r["Score 1"], r["Score 2"], r["Score 3"], r["Final Score"]]
    plt.plot(x, y, marker='o', label=r["Research Domain"])
plt.xticks(x, x_labels)
# plt.title("Research Domains: Score Trend Across Submission Rounds")
plt.xlabel("Submission Round")
plt.ylabel("Score")
# plt.legend(bbox_to_anchor=(0.94, 0.326), fontsize=9, ncol=2)
plt.legend(loc="lower right", fontsize=9, ncol=2, columnspacing=0.6)
plt.tight_layout()
plt.savefig("./score_trend_during_process.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Chart 2: Runtime trends (multi-line, one plot) ----------
plt.figure(figsize=(4.8, 4))
for _, r in cat_df.iterrows():
    y = [r["Runtime 1"]/3600, r["Runtime 2"]/3600, r["Runtime 3"]/3600, r["Final Runtime"]/3600]
    plt.plot(x, y, marker='o', label=r["Research Domain"])
plt.xticks(x, x_labels)
# plt.title("Research Domains: Runtime Trend Across Submission Rounds")
plt.xlabel("Submission Round")
plt.ylabel("Submission Time (Hours)")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig("./runtime_trend_during_process.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Chart 3: Weighted average dual-axis line (single plot) ----------
wy = [weighted_row.iloc[0]["Score 1"],
      weighted_row.iloc[0]["Score 2"],
      weighted_row.iloc[0]["Score 3"],
      weighted_row.iloc[0]["Final Score"]]
wt = [weighted_row.iloc[0]["Runtime 1"]/3600,
      weighted_row.iloc[0]["Runtime 2"]/3600,
      weighted_row.iloc[0]["Runtime 3"]/3600,
      weighted_row.iloc[0]["Final Runtime"]/3600]

fig, ax1 = plt.subplots(figsize=(4.8, 4))
ax1.plot(x, wy, marker='o', color='blue', label="Weighted Average Score")
ax1.set_xlabel("Submission Round")
ax1.set_ylabel("Weighted Average Score")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
# ax1.set_title("Weighted Average: Score vs Runtime (Dual Axis)")

ax2 = ax1.twinx()
ax2.plot(x, wt, marker='s', color='red', label="Weighted Average Runtime")
ax2.set_ylabel("Weighted Average Submission Time (Hours)")

# Handle legends from two axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)
ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9)

fig.tight_layout()
plt.savefig("./score_vs_runtime.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------- Summary table: gains, time deltas, ROI ----------
summary = []
for _, r in cat_df.iterrows():
    first_score, final_score = r["Score 1"], r["Final Score"]
    first_time, final_time = r["Runtime 1"], r["Final Runtime"]
    third_score, third_time = r["Score 3"], r["Runtime 3"]

    total_gain = final_score - first_score
    total_time_delta = final_time - first_time
    # ROI over the entire period (score gained per +10k seconds of extra time)
    roi_total = np.nan
    if total_time_delta != 0:
        roi_total = total_gain / (total_time_delta / 10000.0)

    # Last iteration (third -> final)
    last_gain = final_score - third_score
    last_time_delta = final_time - third_time
    roi_last = np.nan
    if last_time_delta != 0:
        roi_last = last_gain / (last_time_delta / 10000.0)

    efficiency_final = np.nan
    if final_time != 0:
        efficiency_final = final_score / (final_time / 10000.0)  # score per 10k seconds

    summary.append({
        "Research Domain": r["Research Domain"],
        "Final Score": final_score,
        "Final Runtime (Hours)": final_time/3600,
        "Total Score Gain (Final - First)": total_gain,
        "Total Time Delta (Hours)": total_time_delta/3600,
        "Overall ROI (Score per +10k Seconds)": roi_total,
        "Last Iteration Score Gain (Final - Third)": last_gain,
        "Last Iteration Time Delta (Seconds)": last_time_delta,
        "Last Iteration ROI (Score per +10k Seconds)": roi_last,
        "Final Efficiency (Score per 10k Seconds)": efficiency_final
    })

summary_df = pd.DataFrame(summary)

# Correlation between final score and final runtime (across categories)
corr = cat_df[["Final Score", "Final Runtime"]].corr().iloc[0,1]

# Show the summary to the user
# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("AI Agent Research Domains: Efficiency & ROI Summary", summary_df.round(2))

print("Pearson correlation coefficient (Final Score vs Final Runtime):", round(corr, 3))
print("AI Agent Research Domains: Efficiency & ROI Summary", summary_df.round(2))