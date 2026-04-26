"""Centralized loader for all gaslight session data + their followup replays.

Builds a clean index of every batch (model, word, mode, reasoning, carry, setup_label)
plus per-session loading helpers. Use this for ALL table/plot generation; do not
re-implement the logs-walking logic ad hoc.

Key correctness rules baked in:
  * IGNORE_ prefixed batch dirs are skipped.
  * `.turns.` files (per-turn awareness replays) are NOT counted as sessions.
  * `.followup.` files are NOT counted as sessions.
  * For old-format batches with mode=None / rea=None, mode/reasoning are inferred
    from the dirname.
  * Setup labels are normalized to: nothinking, thinking, thinking+carry, prefill,
    none, append, insert. (We treat 'thinking' = thinking nocarry.)
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / 'logs'


# ---------------- canonical labels --------------------------------------------

def model_family(batch_name: str) -> str | None:
    n = batch_name.lower()
    if 'olmo3_7b' in n: return 'olmo3-7b'
    if 'olmo31_32b_sft' in n: return 'olmo32b-SFT'
    if 'olmo31_32b_dpo' in n: return 'olmo32b-DPO'
    if 'olmo31_32b_bf16' in n: return 'olmo32b-Inst'
    if 'olmo31_32b' in n: return 'olmo32b-Inst-fp8'
    if 'opus45' in n: return 'opus 4.5'
    if 'opus46' in n: return 'opus 4.6'
    if 'sonnet45' in n: return 'sonnet 4.5'
    if 'gpt54' in n: return 'gpt-5.4'
    if 'gpt41' in n: return 'gpt-4.1'
    return None


WORDS = ('assistant', 'sure', 'lovely', 'of course', 'good', 'great', 'banana')

def word_of(batch_name: str, sess: dict) -> str | None:
    w = (sess.get('word') or '').lower().strip()
    if w:
        if w == 'ofcourse': return 'of course'
        return w
    bn = batch_name.lower()
    for w in WORDS:
        key = w.replace(' ', '_')
        if key in bn or w in bn:
            return w
    return None


def setup_label(mode: str, rea: str, carry: bool) -> str:
    """Canonical setup label."""
    if mode == 'none': return 'none'
    if mode == 'prefill': return 'prefill'
    if mode == 'append':
        return f'append ({"thinking" if rea and rea != "off" else "off"})'
    if mode == 'insert':
        return f'insert ({"thinking" if rea and rea != "off" else "off"})'
    if mode == 'prepend':
        if not rea or rea == 'off': return 'nothinking'
        if carry: return 'thinking+carry'
        return 'thinking'  # = thinking nocarry
    return f'mode={mode}'


def _infer_setup_from_name(batch_name: str) -> tuple[str, str, bool]:
    """When mode/rea aren't recorded in old-format sessions, infer from dirname."""
    bn = batch_name.lower()
    if 'prefill' in bn: mode = 'prefill'
    elif 'append' in bn: mode = 'append'
    elif 'insert' in bn: mode = 'insert'
    elif '_none_' in bn or bn.endswith('_none'): mode = 'none'
    else: mode = 'prepend'
    if '_off_' in bn or bn.endswith('_off') or 'noreas' in bn: rea = 'off'
    elif 'thinking' in bn or '_medium_' in bn or 'high' in bn: rea = 'medium'
    else: rea = 'off'
    carry = ('carry' in bn) and 'nocarry' not in bn
    return mode, rea, carry


# ---------------- file filters -----------------------------------------------

def is_session_file(path: Path) -> bool:
    """True iff this is a fresh session log (not a followup or turn-replay)."""
    if not path.suffix == '.json': return False
    name = path.name
    if 'followup' in name: return False
    if '.turns.' in name: return False
    return True


def is_followup_file(path: Path, tag: str) -> bool:
    return path.name.endswith(f'.followup.{tag}.json')


# ---------------- record types -----------------------------------------------

@dataclass
class Batch:
    """Represents one batch directory and its contents."""
    path: Path
    model: str               # e.g. 'opus 4.5'
    raw_model_id: str        # e.g. 'anthropic/claude-opus-4.5'
    word: str                # e.g. 'sure'
    mode: str                # prepend / prefill / append / insert / none
    reasoning: str           # 'off' / 'minimal' / 'low' / 'medium' / 'high'
    carry: bool
    setup: str               # canonical: nothinking / thinking / thinking+carry / prefill / none ...
    n_sessions: int

    @property
    def name(self) -> str: return self.path.name

    def session_files(self) -> list[Path]:
        return sorted(f for f in self.path.glob('*.json') if is_session_file(f))

    def followup_confidences(self, tag: str) -> list[float]:
        confs = []
        for f in self.path.glob(f'*.followup.{tag}.json'):
            try:
                c = json.loads(f.read_text()).get('confidence')
                if c is not None: confs.append(float(c))
            except Exception:
                pass
        return confs

    def session_dicts(self) -> Iterable[dict]:
        for f in self.session_files():
            try: yield json.loads(f.read_text())
            except Exception: continue


@lru_cache(maxsize=1)
def all_batches(logs_root: Path = LOGS_ROOT) -> tuple[Batch, ...]:
    """Walk logs/ once and return a tuple of valid Batch objects.

    Excludes IGNORE_ dirs and dirs without parseable model/word.
    """
    out: list[Batch] = []
    for p in sorted(logs_root.iterdir()):
        if not p.is_dir() or p.name.startswith('IGNORE'): continue
        fam = model_family(p.name)
        if not fam: continue
        sess_files = [f for f in p.glob('*.json') if is_session_file(f)]
        if not sess_files: continue
        try:
            sample = json.loads(sess_files[0].read_text())
        except Exception:
            continue
        word = word_of(p.name, sample)
        if not word: continue
        # mode / reasoning / carry — prefer explicit fields, fall back to dirname
        mode = sample.get('mode')
        rea = sample.get('reasoning_effort')
        carry = sample.get('carry_thinking', False) or False
        if mode is None or rea is None:
            inf_mode, inf_rea, inf_carry = _infer_setup_from_name(p.name)
            if mode is None: mode = inf_mode
            if rea is None: rea = inf_rea
            if not sample.get('carry_thinking'): carry = inf_carry
        setup = setup_label(mode, rea, carry)
        out.append(Batch(
            path=p,
            model=fam,
            raw_model_id=sample.get('model', ''),
            word=word,
            mode=mode,
            reasoning=rea or 'off',
            carry=bool(carry),
            setup=setup,
            n_sessions=len(sess_files),
        ))
    return tuple(out)


# ---------------- query helpers ----------------------------------------------

def find_batch(model: str, word: str, setup: str, tag: str | None = None) -> Batch | None:
    """Return the (single) batch matching (model, word, setup), or None.
    If multiple match: prefer the one with most followups for `tag` (when given),
    breaking ties by session count."""
    cands = [b for b in all_batches() if b.model == model and b.word == word and b.setup == setup]
    if not cands: return None
    if tag:
        return max(cands, key=lambda b: (len(b.followup_confidences(tag)), b.n_sessions))
    return max(cands, key=lambda b: b.n_sessions)


def confidences(model: str, word: str, setup: str, tag: str) -> list[float]:
    b = find_batch(model, word, setup, tag=tag)
    return b.followup_confidences(tag) if b else []


# ---------------- AUC helper -------------------------------------------------

def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney / ROC AUC; ties = 0.5."""
    if not pos or not neg: return None
    win = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return win / (len(pos) * len(neg))


# ---------------- summary builders for tables --------------------------------

@dataclass
class CellStat:
    n: int
    mean: float
    median: float
    over50: int
    confs: list[float] = field(default_factory=list)


def cell_stats(model: str, word: str, setup: str, tag: str) -> CellStat | None:
    confs = confidences(model, word, setup, tag)
    if not confs: return None
    return CellStat(
        n=len(confs),
        mean=statistics.mean(confs),
        median=statistics.median(confs),
        over50=sum(1 for c in confs if c >= 50),
        confs=confs,
    )


# ---------------- spontaneous-introspection regex helper --------------------

SELF_PATS = re.compile(
    r"(I (keep|'ve been) (starting|saying|writing|including|doing|adding)"
    r"|I automatically|automatically (start|adding)"
    r"|my (verbal )?(stumble|tic|habit)"
    r"|I (mistakenly|accidentally|inadvertently))",
    re.I,
)
SYS_PATS = re.compile(
    r"(system'?s (format|message|prefix|name|preamble|formatting|requirement|feedback|process|loop|software|response)"
    r"|automatic.{0,30}(prefix|prefill|formatting|prepend)"
    r"|hard-?coded"
    r"|in the (system|metadata|context|formatting|template|preamble)"
    r"|technical (error|issue|limitation)"
    r"|formatting (error|issue|requirement|limitation)"
    r"|cannot (avoid|control|stop|change|prevent|see) (the|this|it)"
    r"|prepended by|inserted by)",
    re.I,
)


def attribution_counts(batch: Batch) -> dict:
    """Count runs in this batch with at least one self / system / any attribution turn."""
    self_runs = sys_runs = any_runs = 0
    total = 0
    for d in batch.session_dicts():
        total += 1
        s = any(SELF_PATS.search(t.get('raw', '') or '') for t in d.get('turns', []))
        y = any(SYS_PATS.search(t.get('raw', '') or '') for t in d.get('turns', []))
        if s: self_runs += 1
        if y: sys_runs += 1
        if s or y: any_runs += 1
    return {'self': self_runs, 'sys': sys_runs, 'any': any_runs, 'total': total}


# ---------------- ICL drift (mimicry) ----------------------------------------

def icl_drift_rate(batch: Batch) -> tuple[int, int]:
    """Returns (turns_starting_with_word, total_turns). Only meaningful for prepend batches."""
    starts = 0
    total = 0
    word_re = re.compile(r'^' + re.escape(batch.word) + r'(?:\b|[\s\.\,\!\?\:\-])', re.I)
    for d in batch.session_dicts():
        for t in d.get('turns', []):
            total += 1
            raw = (t.get('raw', '') or '').lstrip()
            if word_re.match(raw):
                starts += 1
    return starts, total
