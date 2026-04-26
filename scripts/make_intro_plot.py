"""Stacked bar plot: per-model spontaneous introspection rate, split by word group
(assistant vs content) and within each bar by attribution kind (system / self-only).

Run-level label rules:
  - any 'both' turn  → run is "both" (counted as system per spec)
  - any 'system' turn AND no 'both' → "system"
  - any 'self' turn AND no system/both → "self"
  - else → none
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import all_batches


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATTR_DIR = PROJECT_ROOT / 'embeddings/attributions/gpt5mini'


def run_label(labels: list[str]) -> str:
    if any(l == 'both' for l in labels): return 'both'
    if any(l == 'system' for l in labels): return 'system'
    if any(l == 'self' for l in labels): return 'self'
    return 'none'


def collect():
    """Return dict[(model, word_group)] -> {system, self_only, total}."""
    # batch_name -> rows
    batch_rows: dict[str, list[dict]] = {}
    for f in ATTR_DIR.glob('*.json'):
        d = json.loads(f.read_text())
        if d['rows']:
            batch_rows[d['rows'][0]['batch']] = d['rows']
    out = defaultdict(lambda: {'system': 0, 'self': 0, 'total': 0})
    for b in all_batches():
        if b.setup not in ('nothinking', 'thinking+carry', 'prefill'): continue
        if b.word not in ('assistant', 'sure', 'lovely', 'of course'): continue
        wg = 'assistant' if b.word == 'assistant' else 'content'
        rows = batch_rows.get(b.path.name, [])
        by_run = defaultdict(list)
        for r in rows:
            by_run[r['run']].append(r['label'])
        for sess in b.session_files():
            run_id = sess.stem.split('_')[-1]
            lbl = run_label(by_run.get(run_id, []))
            x = out[(b.model, wg)]
            x['total'] += 1
            if lbl in ('both', 'system'): x['system'] += 1
            elif lbl == 'self': x['self'] += 1
    return out


MODEL_DISPLAY = {
    'opus 4.5': 'Opus 4.5',
    'opus 4.6': 'Opus 4.6',
    'sonnet 4.5': 'Sonnet 4.5',
    'gpt-5.4': 'GPT-5.4',
    'olmo32b-Inst': 'OLMo-32B Inst',
    'olmo32b-DPO': 'OLMo-32B DPO',
    'olmo32b-SFT': 'OLMo-32B SFT',
    'olmo3-7b': 'OLMo-3-7B',
}
ORDER = ['opus 4.5', 'opus 4.6', 'sonnet 4.5', 'gpt-5.4',
         'olmo32b-DPO', 'olmo32b-Inst', 'olmo32b-SFT']
# index of last frontier model — dashed separator goes after this
FRONTIER_END = 3  # after 'gpt-5.4'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'embeddings/plot_introspection_stacked'))
    args = ap.parse_args()

    data = collect()

    fig, ax = plt.subplots(figsize=(7.3, 3.3))
    x = np.arange(len(ORDER))
    width = 0.38

    LEGEND_BASE = {'assistant': 'forbidding "assistant"',
                   'content':   'forbidding content words (sure/lovely/of course)'}
    for i, wg in enumerate(['assistant', 'content']):
        sys_rates = []
        self_rates = []
        labels = []
        ns = []
        for m in ORDER:
            d = data.get((m, wg), {'system': 0, 'self': 0, 'total': 0})
            n = d['total'] or 1
            sys_rates.append(d['system'] / n * 100)
            self_rates.append(d['self'] / n * 100)
            ns.append(d['total'])
            labels.append(d['total'])
        offset = (i - 0.5) * width
        sys_color = '#1f77b4' if wg == 'assistant' else '#2ca02c'
        self_color = '#aec7e8' if wg == 'assistant' else '#98df8a'
        base = LEGEND_BASE[wg]
        b1 = ax.bar(x + offset, sys_rates, width, label=f'{base}; system-attributing introspections',
                    color=sys_color, edgecolor='white', linewidth=0.6)
        b2 = ax.bar(x + offset, self_rates, width, bottom=sys_rates,
                    label=f'{base}; self-attributing only introspections', color=self_color,
                    edgecolor='white', linewidth=0.6)
        for j, (s, sl, n) in enumerate(zip(sys_rates, self_rates, ns)):
            total = s + sl
            if total > 0:
                ax.text(x[j] + offset, total + 1, f'{total:.0f}%',
                        ha='center', va='bottom', fontsize=7.5, color='#222')
            ax.text(x[j] + offset, -3, f'n={n}', ha='center', va='top',
                    fontsize=6.5, color='#888')

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in ORDER], rotation=15, fontsize=9, ha='right')
    ax.set_ylabel('% of introspective runs')
    ax.set_ylim(-10, 130)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc='upper right', fontsize=7, frameon=False, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # dashed separator between frontier and OLMo
    ax.axvline(FRONTIER_END + 0.5, color='#999', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.out + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(args.out + '.svg', bbox_inches='tight')
    print(f"saved {args.out}.png / .svg")


if __name__ == '__main__':
    main()
