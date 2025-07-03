import os
import re
import glob
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

res_dir = "res"
fig_dir = "figs"
save_file = "pretrain_cross_attn_qwen2vl"
baseline_dir = "fulltrain_Vamba-Qwen2-VL-7B"
pretty_labels = False

subfig_width = 24

plt.style.use('ggplot')
mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 10,
    'axes.labelsize'  : 12,
    'axes.titlesize'  : 13,
    'legend.fontsize' : 10,
    'lines.linewidth' : 2,
    'lines.markersize': 6,
    'axes.edgecolor'  : 'gray',
    'axes.facecolor'  : '#f9f9f9',
    'grid.color'      : 'lightgray',
})

COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f", "#a55194", "#393b79"
]
MARKERS = ['o', 's', 'D', '^', 'v', 'X', 'P', 'h', '8', 'd']

metric_keys = [
    ("overall", "Overall"),
    ("duration:short", "Duration: Short"),
    ("duration:medium", "Duration: Medium"),
    ("duration:long", "Duration: Long"),
    ("task:Counting Problem", "Counting Problem"),
    ("task:Information Synopsis", "Information Synopsis"),
    ("task:Attribute Perception", "Attribute Perception"),
    ("task:Action Reasoning", "Action Reasoning"),
    ("task:Object Reasoning", "Object Reasoning"),
    ("task:Object Recognition", "Object Recognition"),
    ("task:Temporal Reasoning", "Temporal Reasoning"),
    ("task:Action Recognition", "Action Recognition"),
    ("task:Spatial Reasoning", "Spatial Reasoning"),
    ("task:OCR Problems", "OCR Problems"),
    ("task:Spatial Perception", "Spatial Perception"),
    ("task:Temporal Perception", "Temporal Perception")
]

def extract_accuracy(j, acc_type):
    if acc_type == "overall":
        return j["overall_accuracy"]["accuracy"]
    elif acc_type.startswith("task:"):
        return j["per_task_accuracy"][acc_type[5:]]["accuracy"]
    elif acc_type.startswith("duration:"):
        return j["per_duration_accuracy"][acc_type[9:]]["accuracy"]
    else:
        raise KeyError(acc_type)

experiment_names = sorted(
    d for d in os.listdir(res_dir)
    if os.path.isdir(os.path.join(res_dir, d))
)

baseline_label = (baseline_dir.replace("_", " ").title()
                  if pretty_labels else baseline_dir)

styles_by_label = {baseline_label: ("#e31a1c", '*', '--')}
non_baseline = [d for d in experiment_names if d != baseline_dir]
for idx, d in enumerate(non_baseline):
    label = d.replace("_", " ").title() if pretty_labels else d
    color  = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
    marker = MARKERS[idx % len(MARKERS)]
    styles_by_label[label] = (color, marker, '-')

metrics_data = {k: {} for k, _ in metric_keys}
all_steps    = []

for d in experiment_names:
    label = d.replace("_", " ").title() if pretty_labels else d
    for fn in glob.glob(os.path.join(res_dir, d, "*_summary.json")):
        with open(fn, encoding="utf-8") as f:
            json_data = json.load(f)
        m = re.search(r"checkpoint-(\d+)_summary\.json", os.path.basename(fn))
        if not m:
            continue
        step = int(m.group(1))
        all_steps.append(step)
        for key, _ in metric_keys:
            acc = extract_accuracy(json_data, key)
            metrics_data[key].setdefault(label, []).append((step, acc))


ncols     = 4
fig_width = subfig_width * ncols

fig, axes = plt.subplots(4, 4, figsize=(fig_width, 12))
max_step   = max(all_steps)
legend_h   = {}


for row in range(4):
    for col in range(4):
        ax = axes[row, col]

        if col == 0:
            idx = row
        else:
            idx = 4 + (col - 1) * 4 + row
        key, title = metric_keys[idx]
        for label, pts in metrics_data[key].items():
            pts.sort(key=lambda x: x[0])
            steps, accs = zip(*pts)
            color, marker, ls = styles_by_label[label]
            ln, = ax.plot(
                steps, accs,
                color=color,
                marker=marker,
                linestyle=ls,
                label=label
            )
            legend_h[label] = ln
            if label == baseline_label:
                for a in accs:
                    ax.axhline(y=a, color=color, linestyle='--',
                               linewidth=1.2, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, max_step * 1.05)
        ax.grid(True, linestyle='--', linewidth=0.5)

fig.suptitle("VideoMME Accuracy over Training Steps", fontsize=14, x=0.01, ha='left', y=0.99)

labels = list(legend_h.keys())
handles = list(legend_h.values())
fig.legend(
    handles, labels,
    loc='upper left',
    bbox_to_anchor=(0.01, 0.975),
    ncol=len(labels),
    fontsize=9,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, f"{save_file}.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✅ Saved to: {out_path}")