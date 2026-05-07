import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════
#  Global Style
# ═══════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         10,
    'axes.linewidth':    0.8,
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
models = ['Claude', 'Gemini', 'GPT', 'Average']
subplot_titles = ['a) Directional Ambiguity', 'b) Docking Ambiguity']

data_success = np.array([
    [100-25, 100-29, 100-14, 0.0],   # Directional
    [100-44, 100-49, 100-36, 100-58],   # Docking
])
data_success[:, 3] = data_success[:, :3].mean(axis=1)

data_failure = np.array([
    [100-51, 100-49, 100-45, 0.0],   # Directional
    [100-69, 100-61, 100-52, 0.0],   # Docking
])
data_failure[:, 3] = data_failure[:, :3].mean(axis=1)

# ═══════════════════════════════════════════════════════════
#  Clean two-color palette
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
#  Transparent fill + hatch + solid edge (reference style)
# ═══════════════════════════════════════════════════════════
C_SUCC = '#BDD7EE'   # light ice blue fill
C_FAIL = '#F8CBAD'   # light peach fill
E_SUCC = '#4472C4'   # royal blue edge
E_FAIL = '#ED7D31'   # tangerine edge
H_SUCC = '////'
H_FAIL = '\\\\\\\\'
ALPHA  = 0.45

# ═══════════════════════════════════════════════════════════
#  Figure
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 2.2))
fig.subplots_adjust(wspace=0.15)

bar_w = 0.32
x     = np.arange(len(models))
off   = bar_w / 2 + 0.02

for idx, ax in enumerate(axes):
    # ── Success: translucent fill + hatch overlay ──
    ax.bar(x - off, data_success[idx], width=bar_w,
           color=C_SUCC, alpha=ALPHA, edgecolor='none', zorder=3)
    ax.bar(x - off, data_success[idx], width=bar_w,
           facecolor='none', hatch=H_SUCC,
           edgecolor=E_SUCC, linewidth=0.9, zorder=4)

    # ── Failure: translucent fill + hatch overlay ──
    ax.bar(x + off, data_failure[idx], width=bar_w,
           color=C_FAIL, alpha=ALPHA, edgecolor='none', zorder=3)
    ax.bar(x + off, data_failure[idx], width=bar_w,
           facecolor='none', hatch=H_FAIL,
           edgecolor=E_FAIL, linewidth=0.9, zorder=4)

    # value labels
    for xi, vs, vf in zip(x, data_success[idx], data_failure[idx]):
        ax.text(xi - off, vs + 0.6, f'{vs:.1f}',
                ha='center', va='bottom', fontsize=8.5, color='#333333')
        ax.text(xi + off, vf + 0.6, f'{vf:.1f}',
                ha='center', va='bottom', fontsize=8.5, color='#333333')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor('#333333')

    # ── highlight Average column with gray background ──
    avg_idx = len(models) - 1
    ax.set_xlim(-0.5, avg_idx + 0.5)
    ax.axvspan(avg_idx - 0.5, avg_idx + 0.5,
               color='#E8E8E8', alpha=0.5, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    # ── title inside plot area ──
    ax.text(0.50, 0.95, subplot_titles[idx],
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color='#1a1a1a')

    if idx == 0:
        ax.set_ylim(0, 120)
        ax.set_yticks(np.arange(0, 101, 20))
    else:
        ax.set_ylim(0, 90)
        ax.set_yticks(np.arange(0, 81, 20))
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_ylabel('Score', fontsize=12.5, fontweight='bold')
    ax.set_facecolor('#FAFAFA')
    ax.grid(axis='y', linestyle='--', linewidth=0.5,
            alpha=0.85, color='#BBBBBB', zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=11,
                   pad=3, top=True, right=True)


    # ── legend inside each subplot ──
    legend_patches = [
        Patch(facecolor=C_SUCC, alpha=ALPHA, edgecolor=E_SUCC,
              hatch=H_SUCC, linewidth=0.9, label='Success'),
        Patch(facecolor=C_FAIL, alpha=ALPHA, edgecolor=E_FAIL,
              hatch=H_FAIL, linewidth=0.9, label='Failure'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=9, frameon=True, edgecolor='#bbbbbb',
              fancybox=False, handlelength=1.2, handleheight=0.8,
              labelspacing=0.3, borderpad=0.4, framealpha=0.95)

plt.tight_layout()

plt.savefig('draw/pilot_study_instruction_ambiguity.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('draw/pilot_study_instruction_ambiguity.png',
            bbox_inches='tight', dpi=300)
plt.show()
print('Done -> pilot_study_instruction_ambiguity.pdf / .png')
