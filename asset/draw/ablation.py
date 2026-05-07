import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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

w1_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sr_w1  = [51.3, 70.4, 57.0, 72.8, 67.1, 59.2]
spl_w1 = [46.0, 58.9, 44.7, 56.4, 48.8, 44.3]
osr_w1 = [54.5, 73.9, 67.0, 76.6, 73.6, 68.8]

w2_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
sr_w2  = [63.4, 70.4, 65.3, 66.3, 63.7, 49.9]
spl_w2 = [51.7, 58.9, 53.4, 53.4, 50.0, 37.4]
osr_w2 = [66.5, 73.9, 69.5, 70.4, 68.6, 61.4]

metrics = ["OSR", "SR", "SPL"]
colors      = ["#3498db", "#e74c3c", "#f39c12"]
face_colors = ["#7ec8e3", "#f4a0a8", "#ffd080"]
markers     = ["o", "s", "D"]

subplots_config = [
    {
        "title":    "a) 2D Structure Weight",
        "x_labels": [str(v) for v in w1_values],
        "xlabel":   "Value",
        "ylim":     (40, 85),
        "yticks":   [40, 50, 60, 70, 80],
        "data":     [osr_w1, sr_w1, spl_w1],
        "optimal_x": 3,
        "optimal_y_range": (40, 85),
        "annot_xy": (3, 40), "annot_xytext": (3.4, 45),
        "legend_loc": "upper right",
    },
    {
        "title":    "b) 3D Geometry Weight",
        "x_labels": [str(v) for v in w2_values],
        "xlabel":   "Value",
        "ylim":     (35, 85),
        "data":     [osr_w2, sr_w2, spl_w2],
        "optimal_x": 1,
        "optimal_y_range": (35, 80),
        "annot_xy": (1, 35), "annot_xytext": (1.4, 40),
        "legend_loc": "upper right",
    },
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12, wspace=0.15)

for col, sp in enumerate(subplots_config):
    ax = axes[col]
    x = np.arange(len(sp["x_labels"]))

    for j, (metric, y) in enumerate(zip(metrics, sp["data"])):
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
            label=metric,
        )

    ax.plot(
        [sp["optimal_x"], sp["optimal_x"]],
        list(sp["optimal_y_range"]),
        color='black', linestyle='--', linewidth=1.5, alpha=0.5,
    )
    ax.annotate(
        'Optimal',
        xy=sp["annot_xy"], xytext=sp["annot_xytext"],
        fontsize=12, fontweight='bold',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', color='black', lw=1.5),
    )

    ax.grid(True, linestyle='--', alpha=0.2, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.05, 0.98, sp["title"], transform=ax.transAxes,
            fontsize=20, fontweight='bold', verticalalignment='top')
    ax.set_xticks(x)
    ax.set_xticklabels(sp["x_labels"])
    ax.set_xlabel(sp["xlabel"], fontweight='bold', fontsize=18)
    if col == 0:
        ax.set_ylabel("Performance (%)", fontweight='bold', fontsize=18)
    ax.set_ylim(sp["ylim"])
    if sp.get("yticks"):
        ax.set_yticks(sp["yticks"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.legend(loc=sp["legend_loc"], frameon=True, fancybox=True,
              shadow=True, edgecolor="#cccccc", fontsize=11,
              labelspacing=1, handletextpad=0.6, prop={"size": 12})

plt.savefig('/hpc2hdd/home/yfeng859/yunhengwang/draw/ablation_line.pdf', bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.savefig('/hpc2hdd/home/yfeng859/yunhengwang/draw/ablation_line.png', bbox_inches='tight', pad_inches=0.15, dpi=300, facecolor='white')
plt.show()
print("Saved -> ablation_line.pdf / .png")
