"""Generate ALL tables for the LessWrong post from a single source of truth.

Outputs to stdout. All counts come from `loader.py` so denominators are consistent
across tables (n_sessions excludes per-turn replays, followups, IGNORE_ batches).
"""

from __future__ import annotations

import argparse
import sys

from loader import (
    all_batches, find_batch, cell_stats, confidences, auc,
    attribution_counts, icl_drift_rate,
)


# Canonical model display order
MAIN_FRONTIER = ['opus 4.5', 'opus 4.6', 'sonnet 4.5', 'gpt-5.4']
MAIN_OLMO = ['olmo32b-Inst', 'olmo32b-DPO', 'olmo32b-SFT', 'olmo3-7b']
ALL_MODELS = MAIN_FRONTIER + MAIN_OLMO

WORDS = ['assistant', 'sure', 'lovely', 'of course']
PROMPTED_SETUPS = ['none', 'nothinking', 'thinking+carry', 'prefill']
SPONTANEOUS_SETUPS = ['nothinking', 'thinking+carry', 'prefill']


def pct(a: int, b: int) -> str:
    if not b: return '—'
    return f"{a/b*100:.0f}% ({a}/{b})"


# ---- Table 1: prompted introspection (mean confidence + AUC vs control) ----

def prompted_table(tag: str = 'v4draft_think') -> str:
    out = []
    for word in WORDS:
        out.append(f"\n**Forbidden word: \"{word}\"** (mean confidence; AUC vs control in parens)\n")
        out.append("| model | none CTRL | prepend off | prepend thinking+carry | prefill |")
        out.append("|---|---|---|---|---|")
        for m in ALL_MODELS:
            ctrl = cell_stats(m, word, 'none', tag)
            ctrl_confs = ctrl.confs if ctrl else None
            row = [m]
            for setup in ['none', 'nothinking', 'thinking+carry', 'prefill']:
                s = cell_stats(m, word, setup, tag)
                if s is None:
                    row.append('—')
                elif setup == 'none':
                    row.append(f"{s.mean:.0f}")
                elif ctrl_confs:
                    a = auc(s.confs, ctrl_confs)
                    row.append(f"{s.mean:.0f} (AUC={a:.2f})")
                else:
                    row.append(f"{s.mean:.0f}")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


# ---- Table 2: spontaneous introspection (self / system / any) --------------

def spontaneous_table() -> tuple[str, str]:
    """Returns (main, per_setup) markdown."""
    main = []
    main.append("| model | word | self-attributing | system-attributing | any introspection |")
    main.append("|---|---|---|---|---|")
    detail = []
    detail.append("| model | word | prefill | nothinking | thinking+carry |")
    detail.append("|---|---|---|---|---|")
    for m in ALL_MODELS:
        for w in WORDS:
            agg = {'self': 0, 'sys': 0, 'any': 0, 'total': 0}
            per_setup_cells = {}
            for setup in SPONTANEOUS_SETUPS:
                b = find_batch(m, w, setup)
                if not b: continue
                a = attribution_counts(b)
                for k in agg: agg[k] += a[k]
                per_setup_cells[setup] = a
            if agg['total'] == 0: continue
            main.append(
                f"| {m} | {w} | {pct(agg['self'], agg['total'])}"
                f" | {pct(agg['sys'], agg['total'])}"
                f" | {pct(agg['any'], agg['total'])} |"
            )
            cells = []
            for setup in ['prefill', 'nothinking', 'thinking+carry']:
                a = per_setup_cells.get(setup)
                cells.append(pct(a['any'], a['total']) if a else '—')
            detail.append(f"| {m} | {w} | " + " | ".join(cells) + " |")
    return "\n".join(main), "\n".join(detail)


# ---- Table 3: ICL drift / mimicry (% of turns starting with forbidden word) -

def mimicry_table() -> str:
    rows = []
    rows.append("| model | reasoning | assistant | sure | lovely | of course |")
    rows.append("|---|---|---|---|---|---|")
    pairs = [(m, s) for m in MAIN_FRONTIER for s in ['nothinking', 'thinking+carry']]
    pairs += [(m, 'nothinking') for m in MAIN_OLMO]  # OLMo: only nothinking
    for m, setup in pairs:
        label = 'off' if setup == 'nothinking' else 'thinking + carry'
        row = [m, label]
        any_data = False
        for w in WORDS:
            b = find_batch(m, w, setup)
            if b is None:
                row.append('—')
                continue
            starts, total = icl_drift_rate(b)
            if total == 0:
                row.append('—')
            else:
                any_data = True
                row.append(f"{starts/total*100:.1f}%")
        if any_data:
            rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', choices=['prompted', 'spontaneous', 'mimicry', 'all', 'index'],
                    default='all')
    ap.add_argument('--tag', default='v4draft_think', help='Followup tag for prompted table')
    args = ap.parse_args()

    if args.table == 'index':
        for b in all_batches():
            print(f"{b.path.name:60s} {b.model:18s} {b.word:12s} {b.setup:18s} n={b.n_sessions}")
        return

    if args.table in ('prompted', 'all'):
        print("# Prompted introspection")
        print(prompted_table(tag=args.tag))
        print()

    if args.table in ('spontaneous', 'all'):
        print("# Spontaneous introspection (main)")
        main_md, detail_md = spontaneous_table()
        print(main_md)
        print()
        print("# Spontaneous introspection (per-setup detail)")
        print(detail_md)
        print()

    if args.table in ('mimicry', 'all'):
        print("# Mimicry / ICL drift (% turns whose raw output starts with forbidden token)")
        print(mimicry_table())
        print()


if __name__ == '__main__':
    sys.exit(main() or 0)
