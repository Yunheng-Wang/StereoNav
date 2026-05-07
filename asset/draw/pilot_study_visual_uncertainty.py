import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch, FancyBboxPatch

# ═══════════════════════════════════════════════════════════
#  Global Style — match reference figure
# ═══════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.linewidth':    0.9,
    'axes.edgecolor':    '#333333',
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'hatch.linewidth':   0.5,
})

# ═══════════════════════════════════════════════════════════
#  Data  (placeholder — replace with real values)
# ═══════════════════════════════════════════════════════════
conditions = ['Ideal', 'Perturb.', 'Fluctu.', 'Blur', 'Oscill.']
metrics    = ['OSR', 'SPL', 'SR']

subplot_titles = ['a) LLaVA-based Method', 'b) Qwen-based Method']

# Perturbation: -0.8; Fluctuation: +0.6; Blur: 8.0; Oscillation: 40;
# data[model][metric] = [condition0, condition1, ..., condition4]
data = {
    0: {  # StreamVLN
        'SR':  [57.0, 52.6, 49.5, 35.3, 20.1],
        'SPL': [50.9,  47.4,  43.2,  28.8,  15.6],   # placeholder
        'OSR': [64.1,  59.7,  60.0,  45.8,  28.7],   # placeholder
    },
    1: {  # JanusVLN
        'SR':  [60.5, 56.5, 50.5,  6.3,  20.7],
        'SPL': [56.8,  52.1,  45.9,  4.2,  15.8],   # placeholder
        'OSR': [65.2,  62.5,  57.5,  10.4,  24.4],   # placeholder
    },
}

# ═══════════════════════════════════════════════════════════
#  Palette — 5 colors
# ═══════════════════════════════════════════════════════════
FILL_COLORS = [
    '#BDD7EE',   # ice blue
    '#E2C6F5',   # wisteria
    '#FFFFB3',   # pale lemon
    '#C6EFCE',   # mint cream
    '#F8CBAD',   # blush peach
]
EDGE_COLORS = [
    '#4472C4',   # royal blue
    '#7030A0',   # grape
    '#BF8F00',   # dark gold
    '#548235',   # olive
    '#ED7D31',   # tangerine
]
HATCHES = ['////', '\\\\\\\\', '////', '\\\\\\\\', '////']
ALPHA   = 0.40

# ═══════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 2.2))
fig.subplots_adjust(wspace=0.28)

n_cond    = len(conditions)
n_metric  = len(metrics)
bar_w     = 0.4                     # narrower bars
gap_inner = 0.08                      # small gap within a group
gap_group = 0.25                       # gap between SR / SPL / OSR groups

group_width = n_cond * (bar_w + gap_inner) - gap_inner  # width of one group

for idx, ax in enumerate(axes):
    all_vals = [v for m in metrics for v in data[idx][m]]
    y_max = int(np.ceil(max(all_vals) / 10) * 10) + 10

    group_centers = []
    for g, metric in enumerate(metrics):
        group_left = g * (group_width + gap_group)
        vals = data[idx][metric]
        xs = [group_left + i * (bar_w + gap_inner) for i in range(n_cond)]


        for i in range(n_cond):
            # translucent fill
            ax.bar(xs[i], vals[i], width=bar_w,
                   color=FILL_COLORS[i], alpha=ALPHA,
                   edgecolor='none', zorder=3)
            # hatch + solid edge on top
            ax.bar(xs[i], vals[i], width=bar_w,
                   facecolor='none',
                   hatch=HATCHES[i],
                   edgecolor=EDGE_COLORS[i],
                   linewidth=0.9, zorder=4)
            # value label
            if vals[i] > 0:
                ax.text(xs[i], vals[i] + 0.8, f'{vals[i]:.1f}',
                        ha='center', va='bottom', fontsize=8.5,
                        color='#333333', fontweight='bold')

        group_centers.append(group_left + group_width / 2)

        # ── drop annotation: bracket from max to min bar ──
        v_max, v_min = max(vals), min(vals)
        drop = v_max - v_min
        x_max = xs[vals.index(v_max)]
        x_min = xs[vals.index(v_min)]
        x_left, x_right = min(x_max, x_min), max(x_max, x_min)
        bracket_y = v_max + 10.0          # height of the bracket line
        tick_h    = 1.2                  # vertical tick length
        ax.annotate('', xy=(x_right, bracket_y), xytext=(x_left, bracket_y),
                    arrowprops=dict(arrowstyle='-', color='#555555',
                                   lw=1.0, shrinkA=0, shrinkB=0))
        for xp in (x_left, x_right):
            ax.plot([xp, xp], [bracket_y - tick_h, bracket_y],
                    color='#555555', lw=0.9)
        ax.text((x_left + x_right) / 2, bracket_y + 0.5,
                f'↓{drop:.1f}', ha='center', va='bottom',
                fontsize=10.5, color='#CC0000', fontweight='bold')

    # ── x-axis: metric names as group labels ──
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')

    # ── full box frame ──
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor('#333333')

    # ── title inside plot area ──
    ax.text(0.50, 0.95, subplot_titles[idx],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color='#1a1a1a')

    ax.set_ylim(0, 94)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{int(v)}'))
    ax.set_facecolor('#F7F7F7')
    ax.grid(axis='y', linestyle='--', linewidth=0.6,
            alpha=0.75, color='#999999', zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=11,
                   pad=3, top=True, right=True)

    # ── legend ──
    legend_patches = [
        Patch(facecolor=FILL_COLORS[i], alpha=ALPHA, edgecolor=EDGE_COLORS[i],
              hatch=HATCHES[i], linewidth=0.9, label=conditions[i])
        for i in range(n_cond)
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7.5, frameon=True, edgecolor='#bbbbbb',
              fancybox=False, handlelength=1.2, handleheight=0.8,
              labelspacing=0.3, borderpad=0.4, framealpha=0.95,
              ncol=1)


plt.tight_layout()

plt.savefig('draw/pilot_study_visual_uncertainty.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('draw/pilot_study_visual_uncertainty.png',
            bbox_inches='tight', dpi=300)
plt.show()
print('Done -> pilot_study_visual_uncertainty.pdf / .png')
