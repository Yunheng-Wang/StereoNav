import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.8,
})

# Academic color palette (inspired by Nature-style figures)
C = {
    "llama_vid": "#7FA6C9",   # soft blue (Americas)
    "vicuna":    "#E89B7E",   # soft coral/orange (Oceania)
    "vila":      "#8FBC8F",   # soft green (Asia)
    "llava":     "#B19CD9",   # soft purple (Europe)
    "qwen":      "#E88B8B",   # soft red (Africa)
    "navid":     "#A0785A",   # soft brown
    "new":       "#D4A017",   # golden yellow
    "uninavid":  "#5BA89A",   # teal
}

data = [
    (2024 + 1/12,  37.4, C["llama_vid"], "NaVid"),
    (2024 + 11/12, 47.0, C["vicuna"],    "Uni-NaVid"),
    (2025 + 2/12,  54.0, C["vila"],      "NaVILA"),
    (2025 + 7/12,  56.9, C["llava"],     "StreamVLN"),
    (2025 + 10/12,  55.4, C["qwen"],      "InternVLA"),
    (2025 + 8/12,  65.1, C["llava"],     "CorrectNav"),
    (2025 + 11/12, 64.3, C["qwen"],      "DualVLN"),
    (2026 + 2/12,  60.5, C["qwen"],      "JanusVLN"),
    # (2026 + 3/12,  58.9, C["llava"],     "PROSPECT"),
    (2026 + 3/12,  56.3, C["llava"],     "DecoVLN"),
]

# Data for training data plot (x in millions, y in success rate)
data_training = [
    (0.69, 45.5, C["llama_vid"], "", "p"),
    (1.318, 55.7, C["llama_vid"], "", "*"),
    (0.031, 52.8, C["vicuna"], "", "p"),
    (0.062, 60.8, C["vila"], "", "p"),
    (0.2, 60.5, C["vicuna"], "", "*"),
    (0.32, 64.2, C["vila"], "", "*"),
    (0.45, 50.9, C["llava"], "", "p"),
    (0.81, 56.3, C["llava"], "", "*"),
    (0.011, 24.7, C["navid"], "", "p"),
    (0.0532, 37.4, C["navid"], "", "*"),
    (0.932, 47.7, C["new"], "", "p"),
    (1.376, 54.0, C["new"], "", "*"),
    (0.337, 40.0, C["uninavid"], "", "p"),
    (0.829, 47.0, C["uninavid"], "", "*"),
]

# data_training = [
#     (1.318, 55.7, C["llama_vid"], "streanvln"),
#     (0.2, 60.5, C["vicuna"], "janusvln"),
#     (0.8, 58.2, C["vila"], "internvla"),
# ]

def draw_subplot(ax, show_points=True, show_inset=True, xlabel="Time (Year-Month)", show_trend=True, show_legend=True, data_source=None, xlim=None, xticks=None, xticklabels=None, point_size_scale=1.0, ylim=None):
    if data_source is None:
        data_source = data

    if show_points:
        for item in data_source:
            x, y, c, lbl = item[0], item[1], item[2], item[3]
            marker = item[4] if len(item) > 4 else "o"
            # Size based on performance (success rate)
            s = 350 * point_size_scale
            if marker in ("s", "^", "p", "*"):
                s *= 1.6  # all special markers same size
            ax.scatter(x, y, s=s, color=c, zorder=3, marker=marker,
                       edgecolors="#3a3a3a", linewidths=1, alpha=0.75)
            # Adjust offset based on circle size
            offset = 10 + (s - 350 * point_size_scale) / 100  # Dynamic offset
            if lbl and lbl != "only MP3D data":
                ax.annotate(lbl, xy=(x, y),
                            xytext=(0, offset), textcoords="offset points",
                            fontsize=8.5, ha="center", va="bottom",
                            color="#2a2a2a", fontweight="bold")

    # Trend curve with shadow effect
    if show_trend:
        xs = np.array([d[0] for d in data_source])
        ys = np.array([d[1] for d in data_source])
        coeffs = np.polyfit(xs, ys, 2)
        x_fit = np.linspace(xs.min(), xs.max(), 300)
        y_fit = np.polyval(coeffs, x_fit)

        # Shadow/glow effect
        ax.plot(x_fit, y_fit, color="#e8a5a5", linewidth=5.0, linestyle="-",
                zorder=5, alpha=0.2)
        ax.plot(x_fit, y_fit, color="#e07b7b", linewidth=3.5, linestyle="-",
                zorder=5, alpha=0.35)
        # Main line
        ax.plot(x_fit, y_fit, color="#d85555", linewidth=2.2, linestyle="-",
                zorder=5, alpha=0.8)

    # Tangent line at 2024-01 to show initial growth rate
    if show_trend:
        x_tangent = 2024 + 1/12
        y_tangent = np.polyval(coeffs, x_tangent)
        # Calculate derivative (slope) at this point: dy/dx = 2*a*x + b
        slope = 2 * coeffs[0] * x_tangent + coeffs[1]
        # Draw tangent line (extended)
        x_tang_range = np.array([x_tangent, x_tangent + 1.25])
        y_tang_range = y_tangent + slope * (x_tang_range - x_tangent)
        ax.plot(x_tang_range, y_tang_range, color="#4a90e2", linewidth=2.0,
                linestyle="--", zorder=2, alpha=0.7)

    # Axes
    if xticks is None:
        tick_vals   = [2024 + 1/12, 2025 + 0/12, 2025 + 6/12, 2025 + 10/12, 2026 + 1/12, 2026 + 4/12]
        tick_labels = ["2024-01",   "2025-01",   "2025-06",   "2025-10",    "2026-01",   "2026-04"]
    else:
        tick_vals = xticks
        tick_labels = xticklabels if xticklabels else xticks

    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels, fontsize=11)

    if xlim is None:
        ax.set_xlim(2024, 2026 + 5/12)
    else:
        ax.set_xlim(xlim)

    ax.set_ylim(*(ylim if ylim else (20, 70)))
    ax.set_xlabel(xlabel, fontsize=13.5, labelpad=7, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=13.5, labelpad=7, fontweight="bold")
    ax.tick_params(labelsize=11)

    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.25, color="#d0d0d0")
    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(1.0)

    # Legend
    if show_legend:
        legend_handles = [
            mpatches.Patch(color=C["llama_vid"], label="LLaMA-VID"),
            mpatches.Patch(color=C["vicuna"],    label="Vicuna"),
            mpatches.Patch(color=C["vila"],      label="VILA-1.5"),
            mpatches.Patch(color=C["llava"],     label="LLaVA-NeXT"),
            mpatches.Patch(color=C["qwen"],      label="Qwen2.5-VL"),
        ]
        ax.legend(handles=legend_handles, fontsize=10, frameon=True,
                  framealpha=0.98, edgecolor="#cccccc",
                  loc="upper left", borderpad=0.7, handlelength=1.2)

    # Inset bar chart
    if show_inset:
        ax_ins = ax.inset_axes([0.63, 0.075, 0.34, 0.36])
        bar_vals   = [33.2, 44.5, 56.7, 66.4, 70.2]
        bar_colors = [C["llama_vid"], C["vicuna"], C["vila"], C["llava"], C["qwen"]]
        ax_ins.bar(range(5), bar_vals, color=bar_colors,
                   edgecolor="#555555", linewidth=0.8, width=0.6, alpha=0.75)
        ax_ins.set_xticks([])
        ax_ins.set_xlabel("Performance of Multimodal Models", fontsize=9.5, labelpad=5, color="#4a4a4a")
        for i, v in enumerate(bar_vals):
            ax_ins.text(i, v + 0.8, str(v), ha="center", va="bottom", fontsize=9, color="#4a4a4a")
        ax_ins.set_ylim(0, 80)
        ax_ins.set_ylabel("")
        ax_ins.tick_params(left=False, labelleft=False, labelsize=6)
        ax_ins.set_facecolor("none")
        ax_ins.grid(False)
        ax_ins.spines["top"].set_visible(False)
        ax_ins.spines["left"].set_visible(False)
        ax_ins.spines["right"].set_visible(False)
        ax_ins.spines["bottom"].set_color("#999999")
        ax_ins.spines["bottom"].set_linewidth(1.0)
        ax_ins.text(0.5, 1.08, "MLVU Dev", transform=ax_ins.transAxes,
                    fontsize=12, ha="center", va="top", fontweight="bold", color="#4a4a4a")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Draw the same plot on both subplots
draw_subplot(ax1, ylim=(35, 70))
draw_subplot(ax2, show_points=True, show_inset=False, xlabel="Training Data (M)", show_trend=True, show_legend=False, data_source=data_training, xlim=(-0.05, 1.5), xticks=[0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4], xticklabels=["0", "0.1", "0.2", "0.3", "0.5", "0.7", "1.0", "1.4"], point_size_scale=0.5, ylim=(20, 70))

# Draw arrows from pentagon (base) to star (additional) for each group
arrow_pairs = [
    (0.69, 45.5, 1.318, 55.7, C["llama_vid"], 0.9,   51.2),
    (0.031, 52.8, 0.2, 60.5, C["vicuna"],     0.18,  55.0),
    (0.062, 60.8, 0.32, 64.2, C["vila"],      0.175,  64.3),
    (0.45, 50.9, 0.81, 56.3, C["llava"],      0.60,  55.2),
    (0.011, 24.7, 0.0532, 37.4, C["navid"],   0.12,  30.0),
    (0.932, 47.7, 1.376, 54.0, C["new"],      1.15,  48.5),
    (0.337, 40.0, 0.829, 47.0, C["uninavid"], 0.6,  41.5),
]
for x1, y1, x2, y2, color, tx, ty in arrow_pairs:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="simple,head_width=8,head_length=4,tail_width=3",
                            color=color, alpha=0.7, zorder=2,
                            connectionstyle="arc3,rad=0",
                            shrinkA=14, shrinkB=14)
    ax2.add_patch(arrow)
    gain = y2 - y1
    ax2.text(tx, ty, f"+{gain:.1f}%", fontsize=11, color=color,
             ha="center", va="center", fontweight="bold")


ax2.scatter([], [], marker="p", color="#888888", s=120, edgecolors="#3a3a3a", linewidths=1, alpha=0.75, label="Base Data")
ax2.scatter([], [], marker="*", color="#888888", s=150, edgecolors="#3a3a3a", linewidths=1, alpha=0.75, label="+ Additional Data")
legend_handles = [
    mpatches.Patch(color=C["llama_vid"], label="StreamVLN"),
    mpatches.Patch(color=C["vicuna"],    label="JanusVLN"),
    mpatches.Patch(color=C["vila"],      label="Efficient-VLN"),
    mpatches.Patch(color=C["llava"],     label="DecoVLN"),
    mpatches.Patch(color=C["navid"],     label="NaVid"),
    mpatches.Patch(color=C["new"],       label="NaVILA"),
    mpatches.Patch(color=C["uninavid"],  label="Uni-NaVid"),
]
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=legend_handles + handles, fontsize=10, frameon=True, framealpha=0.98, edgecolor="#cccccc", loc="lower right", borderpad=0.7, handlelength=1.2)

plt.tight_layout()

plt.savefig("/hpc2hdd/home/yfeng859/yunhengwang/draw/intro.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/hpc2hdd/home/yfeng859/yunhengwang/draw/intro.png", dpi=300, bbox_inches="tight")
print("Saved.")
