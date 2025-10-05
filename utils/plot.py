import math
from typing import Dict, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager as fm
#新增加的两行
import os
from glob import glob

def parse_duration_to_hours(duration: str) -> float:
    """Parse a duration string like '1d 4h 28m 2s' into hours as float."""
    total_seconds = 0
    number = 0
    unit = []
    for char in duration + " ":
        if char.isdigit():
            number = number * 10 + int(char)
        elif char.isspace():
            if unit and number:
                u = "".join(unit)
                if u == "d":
                    total_seconds += number * 24 * 3600
                elif u == "h":
                    total_seconds += number * 3600
                elif u == "m":
                    total_seconds += number * 60
                elif u == "s":
                    total_seconds += number
                number = 0
                unit = []
        else:
            # accumulate unit letters (e.g., d/h/m/s)
            if number == 0 and not unit:
                # ignore leading unit letters without number
                continue
            unit.append(char)

    hours = total_seconds / 3600.0
    return hours


def format_hours_label(hours: float) -> str:
    """Return a compact human-readable label from hours float (e.g., '28h 28m')."""
    total_seconds = int(round(hours * 3600))
    days, rem = divmod(total_seconds, 24 * 3600)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)

    parts: List[str] = []
    if days:
        parts.append(f"{days}天")
    if h:
        parts.append(f"{h}小时")
    # if m and days == 0:
    #     parts.append(f"{m}分钟")
    # if s and days == 0 and h == 0:
    #     parts.append(f"{s}秒")
    # if not parts:
    #     parts.append("0秒")
    return " ".join(parts)


def build_dataset() -> List[Dict[str, object]]:
    """Prepare data entries for plotting."""
    ai_avg_hours = parse_duration_to_hours("6h 10m 5s")
    ai_max_hours = parse_duration_to_hours("1d 4h 28m 2s")
    swe_avg_hours = parse_duration_to_hours("1h")

    dataset: List[Dict[str, object]] = [
        {
            "label": "AI 科研创新任务 · 最长",
            "hours": ai_max_hours,
            "steps": 437,
            "category": "AI_MAX",
        },
        {
            "label": "AI 科研创新任务 · 平均",
            "hours": ai_avg_hours,
            "steps": 179.64,
            "category": "AI_AVG",
        },
        {
            "label": "软件工程任务 · 平均",
            "hours": swe_avg_hours,
            "steps": 50,
            "category": "SWE_AVG",
        },
    ]

    # Sort descending by hours for visual emphasis
    dataset.sort(key=lambda d: float(d["hours"]), reverse=True)
    return dataset


def ensure_cjk_fonts() -> None:
    """Ensure Noto CJK fonts are registered and preferred by Matplotlib."""
    # Proactively register system Noto CJK font files to refresh font manager
    candidate_dirs = [
        "/usr/share/fonts/opentype/noto",
        "/usr/share/fonts/truetype/noto",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/.fonts"),
    ]
    font_patterns = [
        "NotoSansCJK-*.ttc",
        "NotoSerifCJK-*.ttc",
        "NotoSansCJK*.otf",
        "NotoSerifCJK*.otf",
    ]
    for base in candidate_dirs:
        if not os.path.isdir(base):
            continue
        for pattern in font_patterns:
            for path in glob(os.path.join(base, pattern)):
                try:
                    fm.fontManager.addfont(path)
                except Exception:
                    pass

    # Prefer CJK fonts first
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    plt.rcParams["font.serif"] = [
        "Noto Serif CJK SC",
        "DejaVu Serif",
    ]
    plt.rcParams["font.monospace"] = [
        "Noto Sans Mono CJK SC",
        "Noto Sans Mono",
        "DejaVu Sans Mono",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Force FontManager to resolve the specified family now
    try:
        fm.findfont("Noto Sans CJK SC", fallback_to_default=False)
    except Exception:
        fm.findfont("Noto Sans CJK SC", fallback_to_default=True)


def apply_apple_like_style() -> None:
    # 👉 关键：把 CJK 字体放到最前，且不要把西文字体放在它们前面
    ensure_cjk_fonts()
    # 下面这些配色/边框保留
    plt.rcParams["axes.edgecolor"] = "#E5E5EA"
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["xtick.color"] = "white"
    plt.rcParams["ytick.color"] = "white"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlepad"] = 14
    plt.rcParams["figure.facecolor"] = "#001226"
    plt.rcParams["axes.facecolor"] = "#001226"



def plot_benchmark_runtime() -> Tuple[plt.Figure, plt.Axes]:
    dataset = build_dataset()

    labels = [str(d["label"]) for d in dataset]
    hours_values = [float(d["hours"]) for d in dataset]
    steps_values = [d["steps"] for d in dataset]
    categories = [str(d["category"]) for d in dataset]

    # Apple system colors
    color_map = {
        "AI_MAX": "#5E5CE6",  # systemPurple
        "AI_AVG": "#0A84FF",  # systemBlue
        "SWE_AVG": "#9A9AA1",  # neutral gray
    }
    colors = [color_map.get(cat, "#9A9AA1") for cat in categories]

    apply_apple_like_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    y_positions = list(range(len(labels)))
    bar_containers = ax.barh(
        y=y_positions,
        width=hours_values,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=1.2,
        height=0.48,
    )

    # Grid and axes aesthetics
    ax.grid(axis="x", color="#001226", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("运行时间（小时）", fontsize=14, labelpad=10)

    # Compute nice x-limit with headroom
    max_hours = max(hours_values)
    ax.set_xlim(0, max_hours * 1.15)

    # Title and subtitle
    ax.set_title("AI 科研创新任务 v.s. 传统软件工程任务", fontsize=24, pad=16)
    # fig.text(
    #     0.125,
    #     0.93,
    #     "平均与最长运行时间对比（含 SWE 基线）",
    #     fontsize=14,
    #     color="#3A3A3C",
    # )

    # Value annotations
    for idx, rect in enumerate(bar_containers):
        hours = hours_values[idx]
        steps = steps_values[idx]
        label_value = format_hours_label(hours)
        # annotation = f"{label_value}  ·  {steps} 步"
        annotation = f"{label_value}"

        x = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2

        # Place annotation just right to the bar end
        ax.text(
            x + max_hours * 0.01,
            y,
            annotation,
            va="center",
            ha="left",
            fontsize=13,
            color="white",
        )

    # Legend
    # legend_handles = [
    #     Patch(facecolor=color_map["AI_AVG"], edgecolor="white", label="AI Benchmark · 平均"),
    #     Patch(facecolor=color_map["AI_MAX"], edgecolor="white", label="AI Benchmark · 最长"),
    #     Patch(facecolor=color_map["SWE_AVG"], edgecolor="white", label="SWE · 平均"),
    # ]
    # legend = ax.legend(
    #     handles=legend_handles,
    #     loc="lower right",
    #     frameon=False,
    #     fontsize=12,
    #     bbox_to_anchor=(1.0, 0.7),
    # )
    # for text in legend.get_texts():
    #     text.set_color("white")

    # Footer note
    fig.text(
        0.125,
        0.02,
        "注：软件工程任务指 SWE-Bench，AI 科研创新任务指 AI Engineer Benchmark。",
        fontsize=11,
        color="white",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    return fig, ax


def save_and_show(output_path: str = "benchmark_runtime_barh.png") -> None:
    # 确保 CJK 字体注册并优先使用
    ensure_cjk_fonts()
    # 解决负号显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    fig, _ = plot_benchmark_runtime()
    fig.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    save_and_show()


