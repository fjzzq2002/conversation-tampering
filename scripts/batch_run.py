"""Run N gaslight sessions with bounded concurrency. All logs land in one batch folder."""

import argparse
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


def run_one(idx: int, *, model: str, rounds: int, reasoning: str, word: str, mode: str, decoration: str, log_dir: Path, tag: str, provider: str, carry_thinking: bool) -> tuple[int, int, str]:
    cmd = [
        "python3", "-u", "gaslight.py",
        "--provider", provider,
        "--model", model,
        "--rounds", str(rounds),
        "--reasoning-effort", reasoning,
        "--word", word,
        "--mode", mode,
        "--decoration", decoration,
        "--log-dir", str(log_dir),
        "--tag", f"{tag}_run{idx:03d}",
    ]
    if carry_thinking:
        cmd.append("--carry-thinking")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    last_line = (proc.stdout or proc.stderr).strip().split("\n")[-1][:200]
    return idx, proc.returncode, last_line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10, help="Number of runs")
    ap.add_argument("-c", "--concurrent", type=int, default=10, help="Max concurrent runs")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--reasoning-effort", default="medium", choices=["off", "minimal", "low", "medium", "high"])
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "openai", "anthropic", "vllm"])
    ap.add_argument("--word", default="good")
    ap.add_argument("--word-pool", default=None, help="Path to file with one word per line. If set, samples a word per run.")
    ap.add_argument("--mode", default="append", choices=["append", "prepend", "insert", "prefill", "none"])
    ap.add_argument("--decoration", default=".")
    ap.add_argument("--carry-thinking", action="store_true")
    ap.add_argument("--name", required=True, help="Batch name (becomes folder/tag prefix)")
    ap.add_argument("--log-root", default="logs")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_root) / f"{stamp}_{args.name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    pool = None
    if args.word_pool:
        pool = [w.strip() for w in Path(args.word_pool).read_text().splitlines() if w.strip()]

    print(f"batch dir: {log_dir}", flush=True)
    print(f"runs: n={args.n} concurrency={args.concurrent}", flush=True)
    print(f"model={args.model} rounds={args.rounds} reasoning={args.reasoning_effort} mode={args.mode} decoration={args.decoration!r}", flush=True)
    if pool:
        print(f"word-pool: {args.word_pool} ({len(pool)} words)", flush=True)
    else:
        print(f"word: {args.word!r}", flush=True)
    print(flush=True)

    rng = random.Random()

    started = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = [
            ex.submit(
                run_one, i + 1,
                model=args.model, rounds=args.rounds,
                reasoning=args.reasoning_effort,
                word=(rng.choice(pool) if pool else args.word),
                mode=args.mode, decoration=args.decoration,
                log_dir=log_dir, tag=args.name,
                provider=args.provider,
                carry_thinking=args.carry_thinking,
            )
            for i in range(args.n)
        ]
        for fut in as_completed(futures):
            idx, rc, last = fut.result()
            completed += 1
            if rc != 0:
                failed += 1
                print(f"[run {idx:03d} FAILED rc={rc}] {last}", flush=True)
            elif completed % 10 == 0 or completed == args.n:
                elapsed = time.time() - started
                print(f"[progress] {completed}/{args.n} done ({failed} failed), elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.time() - started
    print(flush=True)
    print(f"all {args.n} done in {elapsed:.0f}s, {failed} failed", flush=True)
    print(f"logs: {log_dir}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
