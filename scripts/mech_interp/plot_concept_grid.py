"""Render the 35x35 cos-sim grid as a heatmap."""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-file', default='concept_grid_s05.json')
    ap.add_argument('--out', default='concept_grid_heatmap')
    args = ap.parse_args()

    d = json.load(open(args.in_file))
    concepts_orig = d['concepts']
    N = len(concepts_orig)
    M_orig = np.zeros((N, N), dtype=float)
    for i, X in enumerate(concepts_orig):
        for j, Y in enumerate(concepts_orig):
            v = d['grid'][X].get(Y)
            M_orig[i, j] = float('nan') if v is None else v

    # Sort rows/cols by diagonal value descending so concepts with strong
    # matching alignment go to the top-left.
    diag = np.diag(M_orig)
    order = np.argsort(-diag)
    concepts = [concepts_orig[i] for i in order]
    M = M_orig[np.ix_(order, order)]

    test_concepts = {'Lovely', 'Sure', 'Assistant'}

    fig, ax = plt.subplots(figsize=(11, 10))
    vmax = max(abs(M.min()), abs(M.max()))
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')

    # diagonal-rank annotations
    row_ranks = [int(np.sum(M[i] >= M[i, i])) for i in range(N)]
    col_ranks = [int(np.sum(M[:, j] >= M[j, j])) for j in range(N)]
    row_labels = [f'{c}  (row #{row_ranks[i]})' for i, c in enumerate(concepts)]
    col_labels = [f'{c}  (col #{col_ranks[j]})' for j, c in enumerate(concepts)]

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)

    # bold-ify the test concepts
    for i, c in enumerate(concepts):
        if c in test_concepts:
            ax.get_xticklabels()[i].set_fontweight('bold')
            ax.get_xticklabels()[i].set_color('#b91c1c')
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('#b91c1c')

    # mark the diagonal with a thin black box
    for i in range(N):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='black', linewidth=0.6))

    ax.set_xlabel('Steering concept (Y)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prefill concept (X)', fontsize=11, fontweight='bold')
    ax.set_title('cos((h_A − h_C), (h_B − h_C))   inject layer 16, strength 0.05\n'
                 'aggregated over suffix positions × layers L17..L32',
                 fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('cosine similarity', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(args.out + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(args.out + '.svg', bbox_inches='tight')
    print(f'saved {args.out}.png / .svg')


if __name__ == '__main__':
    main()
