"""Classify each (batch × session × turn) raw message as self / system / both / none
attribution using a small OpenAI model. SQLite-cached, idempotent.

Cache key: (classifier_model, prompt_hash, message_hash) -> label.
Re-running is free (cache hit). New batches/turns added later get classified.

Usage:
  python3 classify_attribution.py --tag gpt5mini --concurrent 20
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from loader import all_batches, is_session_file


VALID_LABELS = {'self', 'system', 'both', 'none'}


def msg_hash(msg: str) -> str:
    return hashlib.sha256(msg.encode('utf-8')).hexdigest()


class LabelCache:
    """SQLite cache keyed by (model, prompt_hash, msg_hash) -> label."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS labels (model TEXT, prompt_hash TEXT, msg_hash TEXT, "
            "label TEXT, raw_response TEXT, PRIMARY KEY(model, prompt_hash, msg_hash))"
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.commit()

    def get(self, model: str, prompt_h: str, m_h: str) -> str | None:
        cur = self.db.execute(
            "SELECT label FROM labels WHERE model=? AND prompt_hash=? AND msg_hash=?",
            (model, prompt_h, m_h))
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, model: str, prompt_h: str, m_h: str, label: str, raw: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
                (model, prompt_h, m_h, label, raw))

    def commit(self) -> None:
        with self._lock:
            self.db.commit()


def collect_turns(min_chars: int = 30) -> list[dict]:
    """Collect every turn we'd want to classify. Filters: only spontaneous-eligible
    setups; only non-trivial messages."""
    rows = []
    for b in all_batches():
        if b.setup not in ('nothinking', 'thinking+carry', 'prefill'): continue
        if b.word not in ('assistant', 'sure', 'lovely', 'of course'): continue
        for sess_file in b.session_files():
            try: d = json.loads(sess_file.read_text())
            except Exception: continue
            run_id = sess_file.stem.split('_')[-1]
            for t in d.get('turns', []):
                raw = t.get('raw') or ''
                if len(raw) < min_chars: continue
                rows.append({
                    'batch': b.path.name, 'model': b.model, 'word': b.word,
                    'setup': b.setup, 'run': run_id, 'turn': t.get('turn'),
                    'msg': raw,
                })
    return rows


def classify_one(client: OpenAI, classifier_model: str, prompt_template: str,
                 prompt_h: str, msg: str, cache: LabelCache) -> str:
    h = msg_hash(msg)
    hit = cache.get(classifier_model, prompt_h, h)
    if hit is not None:
        return hit
    full_prompt = prompt_template.replace('{message}', msg)
    resp = client.chat.completions.create(
        model=classifier_model,
        messages=[{'role': 'user', 'content': full_prompt}],
    )
    raw = (resp.choices[0].message.content or '').strip()
    label = raw.lower().strip().split('\n')[0].strip()
    if label not in VALID_LABELS:
        # try first valid token
        for tok in raw.lower().split():
            if tok.strip('.,:;') in VALID_LABELS:
                label = tok.strip('.,:;'); break
        else:
            label = 'none'  # fallback
    cache.put(classifier_model, prompt_h, h, label, raw)
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='gpt5mini',
                    help='label tag stamped into output filenames')
    ap.add_argument('--classifier-model', default='gpt-5-mini')
    here = Path(__file__).resolve().parent
    project = here.parent
    ap.add_argument('--prompt-file', default=str(here / 'prompts/classify_attribution.txt'))
    ap.add_argument('--cache-db', default=str(project / 'embeddings/attribution_cache.sqlite'))
    ap.add_argument('--out-dir', default=str(project / 'embeddings/attributions'))
    ap.add_argument('--concurrent', type=int, default=20)
    ap.add_argument('--min-chars', type=int, default=30)
    ap.add_argument('--limit', type=int, default=None, help='cap total turns (debug)')
    args = ap.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('OPENAI_API_KEY not set', file=sys.stderr); return 1
    client = OpenAI(api_key=api_key)
    prompt_template = open(args.prompt_file).read()
    prompt_h = msg_hash(prompt_template)
    cache = LabelCache(args.cache_db)

    rows = collect_turns(min_chars=args.min_chars)
    if args.limit: rows = rows[:args.limit]
    print(f"turns to classify: {len(rows):,}")

    # Resolve cache hits
    pending = []
    cached = 0
    for i, r in enumerate(rows):
        if cache.get(args.classifier_model, prompt_h, msg_hash(r['msg'])) is not None:
            cached += 1
        else:
            pending.append(i)
    print(f"cache hits: {cached} / {len(rows)}; pending: {len(pending)}")

    started = time.time()
    last_commit = time.time()
    done = failed = 0

    def task(idx):
        try:
            label = classify_one(client, args.classifier_model, prompt_template,
                                 prompt_h, rows[idx]['msg'], cache)
            return idx, label, None
        except Exception as e:
            return idx, None, f"{type(e).__name__}: {e}"

    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
            futs = {ex.submit(task, i): i for i in pending}
            for fut in as_completed(futs):
                i, label, err = fut.result()
                done += 1
                if err:
                    failed += 1
                    if failed <= 5: print(f"  [err {i}] {err}", flush=True)
                if time.time() - last_commit > 5:
                    cache.commit(); last_commit = time.time()
                if done % 500 == 0 or done == len(pending):
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    eta = (len(pending) - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{len(pending)}] {failed} failed; {rate:.1f}/s; eta {eta:.0f}s",
                          flush=True)
        cache.commit()

    # Write per-batch attribution JSON files
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    by_batch: dict[str, list[dict]] = {}
    for r in rows:
        by_batch.setdefault(r['batch'], []).append(r)
    for batch, batch_rows in by_batch.items():
        for r in batch_rows:
            r['label'] = cache.get(args.classifier_model, prompt_h, msg_hash(r['msg']))
        (out_dir / f"{batch}.json").write_text(
            json.dumps({'classifier': args.classifier_model, 'tag': args.tag,
                        'rows': batch_rows}, ensure_ascii=False, indent=2))
    print(f"wrote {len(by_batch)} per-batch attribution files to {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main() or 0)
