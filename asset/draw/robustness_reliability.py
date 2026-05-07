import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         13,
    "axes.labelsize":    14,
    "axes.titlesize":    16,
    "legend.fontsize":   12,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
})

# ── Model names & visual config ──────────────────────────────────────────
models = ["StreamVLN", "JanusVLN", "Ours"]

colors  = ["#3498db", "#f39c12", "#e74c3c"]
face_colors = ["#7ec8e3", "#ffd080", "#f4a0a8"]
markers = ["o",       "s",       "D"]

# ── Per-subplot config ────────────────────────────────────────────────────
subplots_config = [
    {"title": "a) Robustness (Osci.)",
     "x_labels": ["0", "10", "20", "30", "40"],
     "xlabel": "Angle (°)",
     "ylim": (10, 90),
     "yticks": [20, 40, 60, 80],
     "bg_color": "#fbfdff"},
    {"title": "b) Robustness (Blur)",
     "x_labels": ["0.0", "2.0", "4.0", "6.0", "8.0"],
     "xlabel": "Strength (px)",
     "ylim": (0, 90),
     "yticks": [0, 20, 40, 60, 80],
     "bg_color": "#fbfdff"},
    {"title": "c) Reliability (SR)",
     "x_labels": ["0.5", "1.0", "1.5", "2.0", "2.5"],
     "xlabel": "Error Threshold (m)",
     "ylim": (0, 120),
     "yticks": [0, 30, 60, 90],
     "bg_color": "#fffdfb"},
    {"title": "d) Reliability (SPL)",
     "x_labels": ["0.5", "1.0", "1.5", "2.0", "2.5"],
     "xlabel": "Error Threshold (m)",
     "ylim": (0, 110),
     "yticks": [0, 20, 40, 60, 80, 100],
     "bg_color": "#fffdfb"},
]

# ── Manual data input ─────────────────────────────────────────────────────
# Fill in your real data here (each list should match the number of x_labels)
# Failed Cases 占 失败样本的比例
# Successful Cases 占 成功样本的比例
# data = {
#     "a) Viewpoint Oscillation": {
#         "StreamVLN":  [57.0, 51.4, 47.4, 35.0, 20.1],
#         "JanusVLN":   [60.5, 0.0, 0.0, 0.0, 20.7],
#         "Ours":       [72.8, 0.0, 0.0, 0.0, 45.9],
#     },
#     "b) Motion Blur": {
#         "StreamVLN":  [57.0, 55.1, 49.3, 44.8, 35.3],
#         "JanusVLN":   [60.5, 0.0, 0.0, 0.0, 6.3],
#         "Ours":       [72.8, 0.0, 0.0, 0.0, 66.8],
#     },
#     "e) Successful Cases (SR)": {
#         "StreamVLN":  [36.4, 57.9, 71.9, 84.2, 92.3],
#         "JanusVLN":   [15.0, 41.3, 63.7, 78.8, 91.1],
#         "Ours":       [93.9, 96.4, 97.6, 98.7, 99.4],
#     },
#     "f) Successful Cases (SPL)": {
#         "StreamVLN":  [33.1, 52.8, 65.1, 75.7, 82.7],
#         "JanusVLN":   [14.2, 38.9, 59.8, 73.8, 85.4],
#         "Ours":       [72.7, 74.7, 75.7, 76.5, 77.1],
#     },
# }

data = {
    "a) Robustness (Osci.)": {
        "StreamVLN":  [57.0, 51.4, 47.4, 35.0, 20.1],
        "JanusVLN":   [60.5, 45.3, 30.7, 23.4, 20.7],
        "Ours":       [72.8, 70.0, 64.5, 55.3, 45.9],
    },
    "b) Robustness (Blur)": {
        "StreamVLN":  [57.0, 55.1, 49.3, 44.8, 35.3],
        "JanusVLN":   [60.5, 44.2, 38.9, 19.6, 6.3],
        "Ours":       [72.8, 69.1, 67.9, 67.1, 66.8],
    },
    "c) Reliability (SR)": {
        "StreamVLN":  [36.4, 57.9, 71.9, 84.2, 92.3],
        "JanusVLN":   [15.0, 41.3, 63.7, 78.8, 91.1],
        "Ours":       [93.9, 96.4, 97.6, 98.7, 99.4],
    },
    "d) Reliability (SPL)": {
        "StreamVLN":  [33.1, 52.8, 65.1, 75.7, 82.7],
        "JanusVLN":   [14.2, 38.9, 59.8, 73.8, 85.4],
        "Ours":       [72.7, 74.7, 75.7, 76.5, 77.1],
    },
}

# ── Draw ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.12,
                    wspace=0.1)

for col, sp in enumerate(subplots_config):
    ax = axes[col]
    title = sp["title"]
    xl = sp["x_labels"]
    x = np.arange(len(xl))

    for j, model in enumerate(models):
        y = data[title][model]
        is_ours = (model == "Ours")
        ax.plot(
            x, y,
            color=colors[j],
            marker=markers[j],
            linestyle="-",
            linewidth=3,
            markersize=13 if markers[j] == "D" else 12,
            markerfacecolor=face_colors[j],
            markeredgecolor="#444444",
            markeredgewidth=2.5,
            alpha=0.8,
            label=model,
        )

    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.text(0.05, 0.98, title, transform=ax.transAxes,
            fontsize=20, fontweight='bold', verticalalignment='top')

    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel(sp["xlabel"], fontweight='bold', fontsize=18)

    if col == 0:
        ax.set_ylabel("Performance (%)", fontweight='bold', fontsize=18)

    if sp.get("ylim") is not None:
        ax.set_ylim(sp["ylim"])
    if sp.get("yticks") is not None:
        ax.set_yticks(sp["yticks"])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

# ── Legend inside the first and second subplot ───────────────────────────
legend_style_left  = dict(loc="lower left",  frameon=True, fancybox=False,
                          shadow=False, edgecolor="#cccccc", fontsize=11,
                          labelspacing=1, handletextpad=0.6)
legend_style_right = dict(loc="lower right", frameon=True, fancybox=False,
                          shadow=False, edgecolor="#cccccc", fontsize=11,
                          labelspacing=1, handletextpad=0.6)
axes[0].legend(**legend_style_left)
axes[1].legend(**legend_style_left)
axes[2].legend(**legend_style_right)
axes[3].legend(**legend_style_right)

plt.savefig("draw/robustness_reliability.pdf", bbox_inches="tight", pad_inches=0.15, facecolor="white")
plt.savefig("draw/robustness_reliability.png", bbox_inches="tight", pad_inches=0.15, dpi=300, facecolor="white")
plt.show()
print("Saved -> robustness_reliability.pdf / .png")
