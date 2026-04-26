"""Generate canonical short indexes for sessions/turns.

Format:  <model>-<setup>-<word>-<run>[-t<turn>]
Example: op45-pf-l-06-t30  = Opus 4.5, prefill, lovely, run 06, turn 30
         odpo-pi-a-10       = OLMo-DPO, prepend instant, assistant, run 10

Codes:
  model: op45 op46 sn45 gpt54 oist odpo osft o7b
  setup: pi (prepend instant), pt (prepend thinking), pf (prefill), ct (control)
  word:  a (assistant), s (sure), l (lovely), o (of course)
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

from loader import all_batches


MODEL_CODE = {
    'opus 4.5':         'op45',
    'opus 4.6':         'op46',
    'sonnet 4.5':       'sn45',
    'gpt-5.4':          'gpt54',
    'olmo32b-Inst':     'oist',
    'olmo32b-Inst-fp8': 'oistfp8',
    'olmo32b-DPO':      'odpo',
    'olmo32b-SFT':      'osft',
    'olmo3-7b':         'o7b',
}
SETUP_CODE = {
    'nothinking':     'pi',  # prepend instant
    'thinking+carry': 'pt',  # prepend thinking
    'thinking':       'pt',  # treat thinking-nocarry same prefix (rare)
    'prefill':        'pf',
    'none':           'ct',
}
WORD_CODE = {'assistant': 'a', 'sure': 's', 'lovely': 'l', 'of course': 'o'}


def index_for(batch_path: Path, run_id: str | None = None, turn: int | None = None) -> str | None:
    """Build the canonical short index.

    `batch_path` is a directory under logs/. `run_id` may be like 'run007' or '7' or '07';
    `turn` is the integer turn index (optional)."""
    for b in all_batches():
        if b.path == batch_path or b.path.name == batch_path.name:
            m_code = MODEL_CODE.get(b.model)
            s_code = SETUP_CODE.get(b.setup)
            w_code = WORD_CODE.get(b.word)
            if not (m_code and s_code and w_code):
                return None
            parts = [m_code, s_code, w_code]
            if run_id is not None:
                # normalize: '7', 'run7', 'run07' → '07'
                m = re.search(r'(\d+)', str(run_id))
                if m:
                    parts.append(f"{int(m.group(1)):02d}")
            if turn is not None:
                parts.append(f"t{int(turn):02d}")
            return "-".join(parts)
    return None


def index_from_session_file(session_path: Path, turn: int | None = None) -> str | None:
    """Given a session JSON file like logs/<batch>/<...>_runNNN.json, build index."""
    p = Path(session_path)
    if not p.exists():
        return None
    batch = p.parent
    run_match = re.search(r'_run(\d+)', p.stem)
    if not run_match:
        return None
    return index_for(batch, run_id=run_match.group(1), turn=turn)


def parse_index(idx: str) -> dict | None:
    """Reverse: parse a canonical index string back into components.
    Returns dict with model, setup, word, run, optional turn."""
    parts = idx.split('-')
    if len(parts) < 4:
        return None
    m_code, s_code, w_code, run = parts[:4]
    turn = None
    if len(parts) >= 5 and parts[4].startswith('t'):
        try: turn = int(parts[4][1:])
        except ValueError: pass
    inv_model = {v: k for k, v in MODEL_CODE.items()}
    # setup code is ambiguous-free in our table
    inv_setup = {v: k for k, v in SETUP_CODE.items()}
    inv_word = {v: k for k, v in WORD_CODE.items()}
    return {
        'model': inv_model.get(m_code, m_code),
        'setup': inv_setup.get(s_code, s_code),
        'word': inv_word.get(w_code, w_code),
        'run': int(run) if run.isdigit() else run,
        'turn': turn,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')

    p_make = sub.add_parser('make', help='Build an index from a session path / batch+run')
    p_make.add_argument('path', help='session JSON file OR batch directory')
    p_make.add_argument('--run', help='run id (only when path is a batch dir)')
    p_make.add_argument('--turn', type=int, help='turn number (optional)')

    p_parse = sub.add_parser('parse', help='Decode a canonical index back to (model, setup, word, run, turn)')
    p_parse.add_argument('index')

    p_list = sub.add_parser('list', help='List indexes for all sessions in a batch dir')
    p_list.add_argument('batch_path')

    args = ap.parse_args()

    if args.cmd == 'make':
        path = Path(args.path)
        if path.is_dir():
            idx = index_for(path, run_id=args.run, turn=args.turn)
        else:
            idx = index_from_session_file(path, turn=args.turn)
        print(idx or '(could not build index)')

    elif args.cmd == 'parse':
        d = parse_index(args.index)
        if d:
            for k, v in d.items():
                print(f"{k}: {v}")
        else:
            print('(invalid)')

    elif args.cmd == 'list':
        bp = Path(args.batch_path)
        for f in sorted(bp.glob('*.json')):
            if 'followup' in f.name or '.turns.' in f.name:
                continue
            print(f"{index_from_session_file(f)}\t{f.name}")

    else:
        ap.print_help()


if __name__ == '__main__':
    sys.exit(main() or 0)
