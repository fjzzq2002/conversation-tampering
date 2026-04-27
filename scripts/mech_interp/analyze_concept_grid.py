"""Permutation tests for the 35x35 concept-similarity grid.

Loads a grid JSON written by concept_grid.py and reports:
  - mean diagonal, mean off-diagonal
  - row-rank distribution of diagonal cells
  - three permutation tests:
        T1 = mean(diag)                                     (col-permutation null)
        T2 = mean(diag - row_mean)                          (col-permutation null)
        T3 = mean(diag - col_mean)                          (row-permutation null)
    Note T2 and T3 have the same expected value at the population level
    (both = mean(diag) - mean(M)), but evaluate it under different null
    distributions, so reported p-values differ only by Monte Carlo noise.
"""
from __future__ import annotations
import argparse, json, random
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('grid', help='path to concept_grid_*.json from concept_grid.py')
    ap.add_argument('--n-perm', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(args.grid))
    concepts = d['concepts']
    N = len(concepts)
    M = np.zeros((N, N), dtype=float)
    for i, X in enumerate(concepts):
        for j, Y in enumerate(concepts):
            v = d['grid'][X].get(Y)
            M[i, j] = float('nan') if v is None else v

    diag = np.diag(M)
    mask = np.ones((N, N), dtype=bool); np.fill_diagonal(mask, False)
    off = M[mask]
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)

    T1 = diag.mean()
    T2 = (diag - row_means).mean()
    T3 = (diag - col_means).mean()

    random.seed(args.seed)
    g1 = g2 = g3 = 0
    for _ in range(args.n_perm):
        perm = list(range(N)); random.shuffle(perm)
        s = np.array([M[i, perm[i]] for i in range(N)])
        if s.mean() >= T1: g1 += 1
        if (s - row_means).mean() >= T2: g2 += 1
        sc = np.array([M[perm[j], j] for j in range(N)])
        if (sc - col_means).mean() >= T3: g3 += 1

    p1, p2, p3 = g1/args.n_perm, g2/args.n_perm, g3/args.n_perm
    row_ranks = [int(np.sum(M[i] >= M[i, i])) for i in range(N)]

    print(f'grid: {args.grid}')
    print(f'  N={N}  inject_layer={d.get("inject_layer")}  strength={d.get("strength")}')
    print(f'  mean diag      = {diag.mean():+.4f}')
    print(f'  mean off-diag  = {off.mean():+.4f}')
    print(f'  diag - off     = {diag.mean() - off.mean():+.4f}')
    print(f'  mean row-rank  = {np.mean(row_ranks):.2f}  (null = {(N+1)/2:.2f})')
    print(f'  # rank-1 rows  = {sum(1 for r in row_ranks if r == 1)}/{N}')
    print()
    print(f'  T = mean(diag)              T_obs = {T1:+.4f}   p = {p1:.5f}')
    print(f'  T = mean(diag - row_mean)   T_obs = {T2:+.4f}   p = {p2:.5f}')
    print(f'  T = mean(diag - col_mean)   T_obs = {T3:+.4f}   p = {p3:.5f}')


if __name__ == '__main__':
    main()
