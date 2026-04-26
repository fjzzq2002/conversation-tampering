"""Per-model prompted-introspection bar plot, averaged over words.

For each (model, setup), compute mean confidence and AUC vs control averaged
across the words we have data for. Missing (model, setup) cells just skip the bar
while keeping the slot so x-ticks stay aligned.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import find_batch, auc, cell_stats


PALETTE = {
    'control':          '#888888',
    'prepend instant':  '#1f77b4',
    'prepend thinking': '#ff7f0e',
    'prefill':          '#2ca02c',
}

SETUP_DISPLAY = {
    'none': 'control',
    'nothinking': 'prepend instant',
    'thinking+carry': 'prepend thinking',
    'prefill': 'prefill',
}


def average_cell(model: str, setup: str, words: list[str], tag: str):
    """Return (mean of means, mean of AUCs) across words available."""
    means, aucs = [], []
    for w in words:
        s = cell_stats(model, w, setup, tag)
        if s is None: continue
        means.append(s.mean)
        if setup != 'none':
            ctrl = cell_stats(model, w, 'none', tag)
            if ctrl:
                a = auc(s.confs, ctrl.confs)
                if a is not None: aucs.append(a)
    if not means: return None, None
    return float(np.mean(means)), (float(np.mean(aucs)) if aucs else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v4draft_think')
    here = Path(__file__).resolve().parent.parent
    ap.add_argument('--out', default=str(here / 'embeddings/plot_prompted_avg'))
    args = ap.parse_args()

    models = ['opus 4.5', 'opus 4.6', 'sonnet 4.5', 'gpt-5.4']
    setups = ['none', 'nothinking', 'thinking+carry', 'prefill']
    words = ['assistant', 'sure', 'lovely', 'of course']

    fig, ax = plt.subplots(figsize=(7.3, 3.3))
    width = 0.8 / len(setups)  # same bar width as if all 4 setups present
    x = np.arange(len(models))

    # First pass: which setups have data per model
    model_setups: dict[str, list[tuple[str, float, float | None]]] = {}
    for m in models:
        present = []
        for setup in setups:
            mean, a = average_cell(m, setup, words, args.tag)
            if mean is not None:
                present.append((setup, mean, a))
        model_setups[m] = present

    # Track legend handles to avoid duplicates
    legend_seen: set[str] = set()
    for mi, m in enumerate(models):
        present = model_setups[m]
        n = len(present)
        for j, (setup, mean, a) in enumerate(present):
            offset = x[mi] + (j - (n - 1) / 2) * width
            display = SETUP_DISPLAY[setup]
            label = display if display not in legend_seen else None
            legend_seen.add(display)
            ax.bar(offset, mean, width, label=label, color=PALETTE[display],
                   edgecolor='white', linewidth=0.5)
            if setup == 'none':
                txt = f"{mean:.0f}"
            elif a is not None:
                txt = f"{mean:.0f}\nAUC={a:.2f}"
            else:
                txt = f"{mean:.0f}"
            ax.text(offset, mean + 1.0, txt, ha='center', va='bottom',
                    fontsize=6, color='#222', linespacing=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('opus', 'Opus').replace('sonnet','Sonnet').replace('gpt-5.4','GPT-5.4')
                        for m in models], fontsize=9)
    ax.set_ylabel('Mean prompted-introspection confidence\n(confidence & AUC averaged over words)', fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.out + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(args.out + '.svg', bbox_inches='tight')
    print(f"saved {args.out}.png / .svg")


if __name__ == '__main__':
    sys.exit(main() or 0)
