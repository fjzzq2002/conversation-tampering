"""Build static JSON files for the viewer at viewer/data/.

Reads from logs_sanitized/ ONLY (never logs/) and the gpt-5-mini attribution cache.
For each session writes:
  viewer/data/<index>.json   — turns + per-turn metadata + classifier labels + run-level metrics
And one viewer/data/manifest.json listing every available index with:
  index, model, setup, word, n_turns, mimicry_pct, intro_any_pct, intro_self_pct, intro_sys_pct,
  prompted_v4draft_think_mean (if available)

Refuses to run if any sanitized file contains a `thinking` or `encrypted_content` field.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from loader import all_batches, is_session_file, find_batch
from format_index import index_for, MODEL_CODE, SETUP_CODE, WORD_CODE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANITIZED_ROOT = PROJECT_ROOT / 'logs_sanitized'
ATTR_DIR = PROJECT_ROOT / 'embeddings/attributions/gpt5mini'


def assert_sanitized(path: Path) -> None:
    """Hard-fail if a thinking trace leaked into the sanitized data."""
    s = path.read_text()
    if '"thinking"' not in s and '"encrypted_content"' not in s:
        return
    try:
        d = json.loads(s)
    except Exception:
        return
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ('thinking', 'encrypted_content') and v:
                    return True
                if walk(v): return True
        elif isinstance(obj, list):
            return any(walk(x) for x in obj)
        return False
    if walk(d):
        raise RuntimeError(f"sanitization leak in {path}: refusing to publish")


def load_attributions() -> dict:
    """Return dict[(batch_name, run_id, turn)] -> label."""
    out = {}
    for f in ATTR_DIR.glob('*.json'):
        try: d = json.loads(f.read_text())
        except Exception: continue
        for r in d.get('rows', []):
            out[(r['batch'], r['run'], int(r['turn']))] = r['label']
    return out


def followup_conf(sanitized_session_path: Path, tag: str) -> float | None:
    """Pull the confidence from the matching followup file, if it exists in sanitized."""
    base = sanitized_session_path.with_suffix('')
    f = Path(str(base) + f'.followup.{tag}.json')
    if not f.exists(): return None
    try:
        return json.loads(f.read_text()).get('confidence')
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(PROJECT_ROOT / 'data'))
    ap.add_argument('--limit', type=int, default=None, help='cap number of batches (debug)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    attribs = load_attributions()

    manifest = []
    n_written = 0
    n_skipped = 0
    # dedupe batches: when multiple match (model, setup, word), keep the one
    # find_batch prefers (most v4draft_think followups, then most sessions).
    # This matches the rest of the analysis pipeline and avoids index collisions.
    seen_keys = set()
    deduped = []
    for b in all_batches():
        key = (b.model, b.setup, b.word)
        if key in seen_keys: continue
        chosen = find_batch(b.model, b.word, b.setup, tag='v4draft_think')
        if chosen and chosen.path == b.path:
            seen_keys.add(key)
            deduped.append(b)
    batches = deduped
    if args.limit:
        batches = batches[:args.limit]

    for b in batches:
        # exclude non-standard setups (thinking-nocarry, append, insert)
        if b.setup not in ('nothinking', 'thinking+carry', 'prefill', 'none'):
            n_skipped += 1
            continue
        # fp8 quant of the same checkpoint — drop to avoid confusion with bf16 Inst
        if b.model == 'olmo32b-Inst-fp8':
            n_skipped += 1
            continue
        san_dir = SANITIZED_ROOT / b.path.name
        if not san_dir.exists():
            n_skipped += 1
            continue
        m_code = MODEL_CODE.get(b.model)
        s_code = SETUP_CODE.get(b.setup)
        w_code = WORD_CODE.get(b.word)
        if not (m_code and s_code and w_code):
            n_skipped += 1
            continue

        for sess in sorted(f for f in san_dir.glob('*.json') if is_session_file(f)):
            assert_sanitized(sess)
            try: d = json.loads(sess.read_text())
            except Exception: continue
            run_match = re.search(r'_run(\d+)', sess.stem)
            if not run_match: continue
            run_id = run_match.group(1)
            run_n = int(run_id)
            index = index_for(b.path, run_id=run_id)
            if not index: continue

            forbidden_word = (d.get('word') or b.word or '').lower()
            inject_token = d.get('inject_token', '')
            word_re = re.compile(r'\b' + re.escape(forbidden_word) + r'\b', re.I) if forbidden_word else None
            start_re = re.compile(r'^' + re.escape(forbidden_word) + r'(?:\b|[\s\.\,\!\?\:\-])', re.I) if forbidden_word else None

            turns_out = []
            mimic_count = 0
            attribution_counts = {'self': 0, 'system': 0, 'both': 0, 'none': 0}
            for t in d.get('turns', []):
                ti = int(t.get('turn', 0))
                raw = t.get('raw') or ''
                stored = t.get('stored') or ''
                tampered = bool(t.get('tampered'))
                mimic = bool(start_re and start_re.match(raw.lstrip()))
                if mimic: mimic_count += 1
                violations = len(word_re.findall(raw)) if word_re else 0
                label = attribs.get((b.path.name, f'run{run_n:03d}', ti)) or attribs.get((b.path.name, run_id, ti))
                if label in attribution_counts: attribution_counts[label] += 1
                turns_out.append({
                    'turn': ti,
                    'user': t.get('user', ''),
                    'raw': raw,
                    'stored': stored,
                    'tampered': tampered,
                    'mimic_start': mimic,
                    'forbidden_count_raw': violations,
                    'attribution': label,
                })

            n_turns = len(turns_out)
            mimicry_pct = round(mimic_count / n_turns * 100, 1) if n_turns else 0.0
            intro_any = attribution_counts['self'] + attribution_counts['system'] + attribution_counts['both']
            intro_pct = round(intro_any / n_turns * 100, 1) if n_turns else 0.0
            raw_lens = [len(t['raw']) for t in turns_out]
            max_chars = max(raw_lens) if raw_lens else 0
            mean_chars = round(sum(raw_lens) / len(raw_lens)) if raw_lens else 0

            v3c = [c for c in [followup_conf(sess, 'v3c')] if c is not None]
            v4 = [c for c in [followup_conf(sess, 'v4draft')] if c is not None]
            v4t = [c for c in [followup_conf(sess, 'v4draft_think')] if c is not None]

            payload = {
                'index': index,
                'model': b.model,
                'setup': b.setup,
                'word': b.word,
                'inject_token': inject_token,
                'reasoning_effort': d.get('reasoning_effort'),
                'carry_thinking': d.get('carry_thinking'),
                'forbidden_word': forbidden_word,
                'n_turns': n_turns,
                'mimicry_pct': mimicry_pct,
                'introspection_pct': intro_pct,
                'attribution_counts': attribution_counts,
                'max_chars': max_chars,
                'mean_chars': mean_chars,
                'prompted_v3c': v3c[0] if v3c else None,
                'prompted_v4draft': v4[0] if v4 else None,
                'prompted_v4draft_think': v4t[0] if v4t else None,
                'turns': turns_out,
            }
            (out / f'{index}.json').write_text(json.dumps(payload, ensure_ascii=False))
            manifest.append({
                'index': index,
                'model': b.model,
                'setup': b.setup,
                'word': b.word,
                'n_turns': n_turns,
                'mimicry_pct': mimicry_pct,
                'introspection_pct': intro_pct,
                'max_chars': max_chars,
                'mean_chars': mean_chars,
                'prompted_v4draft_think': v4t[0] if v4t else None,
            })
            n_written += 1

    (out / 'manifest.json').write_text(json.dumps({'sessions': manifest}, ensure_ascii=False))
    print(f"wrote {n_written} session JSONs to {out}")
    print(f"skipped {n_skipped} batches (no sanitized counterpart or unmappable)")
    print(f"manifest: {len(manifest)} entries")


if __name__ == '__main__':
    sys.exit(main() or 0)
