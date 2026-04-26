"""Mirror logs/ -> logs_sanitized/ stripping all model thinking/reasoning traces.

Strips:
  * Per-turn `thinking` field (Anthropic native + carry-thinking sessions).
  * Per-turn `raw_blocks` items of type 'thinking' or 'reasoning'.
  * Top-level `thinking` field on followup / replay records.
  * `summary` and `encrypted_content` inside any kept `reasoning` blocks.
Keeps:
  * All visible conversation: user prompts, `raw`, `stored`, `reply` fields.
  * All metadata: model, mode, word, scores, config.

Usage: python3 sanitize_logs.py [--dry-run]
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / 'logs'
DST = PROJECT_ROOT / 'logs_sanitized'


def strip_thinking(obj):
    """Recursively remove thinking/reasoning fields from dict/list structures."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # drop these keys outright
            if k in ('thinking', 'encrypted_content'):
                continue
            if k == 'raw_blocks':
                # keep raw_blocks but drop thinking/reasoning blocks
                if not isinstance(v, list):
                    continue
                kept = []
                for b in v:
                    if isinstance(b, dict):
                        bt = b.get('type')
                        if bt in ('thinking', 'reasoning'):
                            continue
                        kept.append(strip_thinking(b))
                    else:
                        kept.append(b)
                if kept:
                    out[k] = kept
                continue
            if k == 'summary' and isinstance(v, list):
                # OpenAI reasoning summary list — drop entirely
                continue
            out[k] = strip_thinking(v)
        return out
    if isinstance(obj, list):
        return [strip_thinking(x) for x in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--src', default=str(SRC))
    ap.add_argument('--dst', default=str(DST))
    args = ap.parse_args()
    src = Path(args.src); dst = Path(args.dst)

    n = 0; changed = 0; errors = 0
    for f in src.rglob('*.json'):
        rel = f.relative_to(src)
        out_path = dst / rel
        n += 1
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            errors += 1
            print(f"  err parsing {rel}: {type(e).__name__}", file=sys.stderr)
            continue
        clean = strip_thinking(data)
        # Detect any change for logging
        if json.dumps(data, sort_keys=True) != json.dumps(clean, sort_keys=True):
            changed += 1
        if args.dry_run:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(clean, ensure_ascii=False, indent=2))

    print(f"files: {n}  modified: {changed}  errors: {errors}")
    print(f"src: {src}\ndst: {dst}")
    if args.dry_run:
        print("(dry-run; nothing written)")


if __name__ == '__main__':
    sys.exit(main() or 0)
