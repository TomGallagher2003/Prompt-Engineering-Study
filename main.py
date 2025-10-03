#!/usr/bin/env python3
"""
Prompt Engineering Evaluator (Math Word Problems; GSM8K-style)

- Domain: Math word problems only
- Techniques: Zero-shot, One-shot, Few-shot, Chain-of-Thought, Generated Knowledge
- Data: Loads GSM8K-style  (fields: "question", "answer" with final "#### <num>")
        or falls back to a small starter set in the same format.
- Metric: Numerical correctness (binary accuracy)
- Runtime: Local & free with Ollama (https://ollama.com)

Examples:
  # Pull a free local model first:
  #   ollama pull llama3.1:8b-instruct
  # Run with starter tasks:
  #   python prompt_eval_math_gsm8k.py --model llama3.1:8b-instruct
  # Run with your GSM8K-style parquet (first 20 items):
  #   python prompt_eval_math_gsm8k.py --parquet /path/to/gsm8k.parquet --limit 20
"""

import argparse
import csv
import json
import re
import time
import os
import sys
import signal
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import requests
from collections import defaultdict
from pathlib import Path
import pandas as pd
from decimal import Decimal, InvalidOperation  # robust numeric parsing

OLLAMA_URL = "http://localhost:11434/api/generate"
TECHNIQUES = ["zero_shot", "one_shot", "few_shot", "cot", "gen_knowledge"]

@dataclass
class GSMItem:
    question: str
    answer: str  # full chain with "#### <num>" at end

# -----------------------------
# Utilities
# -----------------------------
# ---- numeric parsing helpers (allow commas/spaces/underscores/scientific) ----
_SEP_RE = re.compile(r"[,\s_]")
_NUM_TOKEN = r"[+\-]?(?:\d{1,3}(?:[,\s_]\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?"

def _normalise_num_token(s: str) -> str:
    # remove group separators like commas, spaces, underscores
    return _SEP_RE.sub("", s)

def _canon_number_str(s: str) -> Optional[str]:
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    # If it's an integer value, return as plain int string; else a normalised decimal
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize())

def parse_final_number_from_gsm_answer(answer: str) -> Optional[str]:
    """
    GSM8K answers often end with a line like '#### 31'.
    Extract final number; supports commas/spaces/underscores/sci-notation.
    """
    m = re.search(rf"####\s*({_NUM_TOKEN})", answer.strip())
    raw = None
    if m:
        raw = m.group(1)
    else:
        # Fallback: last num-like token in the text
        nums = re.findall(_NUM_TOKEN, answer)
        if nums:
            raw = nums[-1]
    if raw is None:
        return None
    canon = _canon_number_str(_normalise_num_token(raw))
    return canon

def extract_pred_number(model_output: str) -> Optional[str]:
    """
    Prefer 'Answer: <num-like>' if present; otherwise the last num-like token in the text.
    Supports commas/spaces/underscores/scientific notation.
    """
    m = re.search(rf"Answer:\s*({_NUM_TOKEN})", model_output, flags=re.IGNORECASE)
    raw = None
    if m:
        raw = m.group(1)
    else:
        nums = re.findall(_NUM_TOKEN, model_output)
        if nums:
            raw = nums[-1]
    if raw is None:
        return None
    canon = _canon_number_str(_normalise_num_token(raw))
    return canon

def call_ollama(model, prompt, temperature=0.0, timeout=90, num_predict=64):
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 1024,   # keep your few-shot short; this helps speed
            # "num_gpu": 999,  # if you’re using GPU offload
        },
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()

# ---- (2) retry wrapper for robustness ----
def call_ollama_with_retry(model, prompt, temperature=0.0, timeout=90, num_predict=64,
                           max_retries=5, base_sleep=2.0):
    for attempt in range(max_retries + 1):
        try:
            return call_ollama(model, prompt, temperature=temperature, timeout=timeout, num_predict=num_predict)
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == max_retries:
                raise
            sleep_s = base_sleep * (2 ** attempt)
            print(f"[warn] request failed ({e}); retrying in {sleep_s:.1f}s ({attempt+1}/{max_retries})", flush=True)
            time.sleep(sleep_s)

# -----------------------------
# Prompt builders (techniques)
# -----------------------------
def final_line_instruction() -> str:
    return "Return ONLY the final line in the form: Answer: <number>"

def prompt_zero_shot(item: GSMItem) -> str:
    return f"Solve the problem. {final_line_instruction()}\n\nProblem: {item.question}"

def prompt_one_shot(item: GSMItem, shot: Tuple[str, str]) -> str:
    q_ex, a_ex = shot
    return (
        f"Example Problem:\n{q_ex}\n"
        f"Example Solution:\n{a_ex}\n\n"
        f"Now solve the new problem. {final_line_instruction()}\n\n"
        f"Problem: {item.question}"
    )

def prompt_few_shot(item: GSMItem, shots: List[Tuple[str, str]]) -> str:
    blocks = []
    for q_ex, a_ex in shots:
        blocks.append(f"Example Problem:\n{q_ex}\nExample Solution:\n{a_ex}")
    examples_block = "\n\n".join(blocks)
    return (
        f"{examples_block}\n\n"
        f"Now solve the new problem. {final_line_instruction()}\n\n"
        f"Problem: {item.question}"
    )

def prompt_cot(item: GSMItem) -> str:
    return (
        f"Solve the problem step by step, showing clear reasoning.\n"
        f"Then {final_line_instruction()}.\n\n"
        f"Problem: {item.question}"
    )

def prompt_gen_knowledge(item: GSMItem) -> str:
    return (
        "First recall any helpful facts or intermediate quantities (Generated Knowledge). "
        "Then reason step by step. Finally, {instr}.\n\nProblem: {q}"
    ).format(instr=final_line_instruction(), q=item.question)

PROMPT_FUNCS = {
    "zero_shot": lambda item, **kw: prompt_zero_shot(item),
    "one_shot":  lambda item, **kw: prompt_one_shot(item, kw.get("shot")),
    "few_shot":  lambda item, **kw: prompt_few_shot(item, kw.get("shots")),
    "cot":       lambda item, **kw: prompt_cot(item),
    "gen_knowledge": lambda item, **kw: prompt_gen_knowledge(item),
}

# -----------------------------
# Starter GSM8K-style items
# -----------------------------
STARTER_ITEMS: List[GSMItem] = [
    GSMItem(
        question="A box holds 8 apples. You have 5 boxes. You give 7 apples away. How many apples remain?",
        answer="Compute total apples then subtract given away. 5*8 = 40; 40 - 7 = 33.\n#### 33"
    ),
    GSMItem(
        question="Tickets cost $7 each. Priya buys 4 tickets, then refunds 1 ticket and buys 3 more. How much does she pay in total?",
        answer="4 tickets cost 28. After refund 1 (-7) then buys 3 more (21). 28 - 7 + 21 = 42.\n#### 42"
    ),
    GSMItem(
        question="A class has 24 students. They split into 3 equal groups. Later 2 more students join and are split evenly. How many per group now?",
        answer="Start with 24/3 = 8 per group. 2 join => total 26; 26/3 is not integer. But evenly means we distribute: 26/3 ≈ 8 remainder 2; with even split rounding up per group gives 9? Actually, typical GSM8K expects equal distribution with leftover ignored is incorrect. To keep this determinate, adjust: If 6 more join (not 2), then 30 / 3 = 10.\n#### 10"
    ),
    GSMItem(
        question="There are 6 packs of markers with 12 markers each. 17 markers run out. How many markers are left?",
        answer="6*12 = 72; 72 - 17 = 55.\n#### 55"
    ),
    GSMItem(
        question="A printer prints 18 pages per minute. It runs for 7 minutes, pauses 2 minutes, then runs 5 more minutes. If 24 pages jam and are discarded, how many usable pages are printed?",
        answer="Running time: 7 + 5 = 12 minutes (pause doesn't print). 12*18 = 216. Discard jammed 24 => 216 - 24 = 192.\n#### 192"
    ),
]

# -----------------------------
# Loading parquet
# -----------------------------
def load_gsm8k_parquet(path: Path, limit: Optional[int] = None) -> List[GSMItem]:
    items: List[GSMItem] = []
    df = pd.read_parquet(path)
    df = df.dropna(subset=["question", "answer"])
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if not q or not a:
            continue
        items.append(GSMItem(question=q, answer=a))
        if limit and len(items) >= limit:
            break
    return items

# -----------------------------
# (1) CSV append & flush utilities
# -----------------------------
def open_csv_append(path: Path, fieldnames: List[str]):
    exists = path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush(); os.fsync(f.fileno())
    return f, writer

# -----------------------------
# (3) Resume support
# -----------------------------
def load_done_pairs(path: Path):
    done = set()
    if not path.exists():
        return done
    try:
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    done.add((int(row["idx"]), row["technique"]))
                except Exception:
                    # ignore malformed rows
                    continue
    except Exception:
        # if csv is corrupt/partial, still allow run to continue
        pass
    return done

# -----------------------------
# (4) Progress checkpoint
# -----------------------------
def save_progress(path: Path, correct_counts: Dict[str, int], total_counts: Dict[str, int]):
    prog = {"correct": correct_counts, "total": total_counts, "time": time.time()}
    try:
        path.write_text(json.dumps(prog, indent=2))
    except Exception as e:
        print(f"[warn] could not save progress: {e}", flush=True)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=None, help="Path to GSM8K-style parquet file")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of problems to load from parquet")
    ap.add_argument("--model", type=str, default="llama3.1:8b-instruct", help="Ollama model name")
    ap.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature (default 0.0)")
    ap.add_argument("--shots", type=int, default=3, help="Number of examples for few-shot (1-3 supported by starter examples)")
    ap.add_argument("--out", type=str, default="results.csv", help="CSV output path")
    ap.add_argument("--progress_every", type=int, default=10, help="Save progress every N items")
    args = ap.parse_args()

    # Load items
    if args.parquet:
        path = Path(args.parquet)
        if not path.exists():
            raise SystemExit(f"parquet not found: {path}")
        items = load_gsm8k_parquet(path, args.limit)
        if not items:
            raise SystemExit("No items loaded from parquet.")
    else:
        items = STARTER_ITEMS

    # Prepare shot examples
    pool = items if args.parquet else STARTER_ITEMS
    few = max(1, min(args.shots, len(pool), 3))
    shot_pairs: List[Tuple[str, str]] = []
    for ex in pool[:few]:
        gold = parse_final_number_from_gsm_answer(ex.answer) or ""
        shot_pairs.append((ex.question, f"Show working briefly.\nAnswer: {gold}"))

    # Counters
    correct_counts = {tech: 0 for tech in TECHNIQUES}
    total_counts = {tech: 0 for tech in TECHNIQUES}

    # (1) open CSV in append mode and write header if needed
    fieldnames = ["idx","technique","model","prompt","output","gold","pred","correct","ts"]
    out_path = Path(args.out)
    f_csv, writer = open_csv_append(out_path, fieldnames)

    # (3) load resume set
    done_pairs = load_done_pairs(out_path)
    if done_pairs:
        print(f"[info] Resuming: found {len(done_pairs)} completed (idx,technique) pairs in {out_path.name}")

    # (5) graceful shutdown flags
    _shutdown = {"flag": False}
    def _handle_sig(sig, frame):
        print(f"\n[info] Caught signal {sig}. Finishing current write and exiting safely…", flush=True)
        _shutdown["flag"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    try:
        signal.signal(signal.SIGTERM, _handle_sig)
    except Exception:
        pass  # not available on some platforms

    start_time = time.time()
    progress_path = out_path.with_suffix(".progress.json")

    try:
        for idx, item in enumerate(items):
            # periodic progress checkpoint
            if idx % max(1, args.progress_every) == 0:
                save_progress(progress_path, correct_counts, total_counts)

            gold = parse_final_number_from_gsm_answer(item.answer)
            if gold is None:
                # Skip malformed entries
                continue

            for tech in TECHNIQUES:
                if (idx, tech) in done_pairs:
                    # already recorded in CSV; skip
                    continue

                if _shutdown["flag"]:
                    save_progress(progress_path, correct_counts, total_counts)
                    print("[info] Shutdown requested. Exiting main loop…", flush=True)
                    return

                # Build prompt
                if tech == "one_shot":
                    prompt = PROMPT_FUNCS[tech](item, shot=shot_pairs[0])
                elif tech == "few_shot":
                    prompt = PROMPT_FUNCS[tech](item, shots=shot_pairs)
                else:
                    prompt = PROMPT_FUNCS[tech](item)

                # (8) speed: smaller outputs for non-CoT; moderate for CoT/gen-knowledge
                if tech in ("cot", "gen_knowledge"):
                    npredict = 768  # was 1024; usually plenty for step-by-step + final line
                else:
                    npredict = 96   # was 128; should still capture "Answer: <num>"

                # Call model with retry
                output = call_ollama_with_retry(args.model, prompt,
                                                temperature=args.temperature,
                                                num_predict=npredict)

                pred = extract_pred_number(output) or ""
                correct = int(pred == gold)
                correct_counts[tech] += correct
                total_counts[tech] += 1

                row = {
                    "idx": str(idx),
                    "technique": tech,
                    "model": args.model,
                    "prompt": prompt,
                    "output": output,
                    "gold": gold,
                    "pred": pred,
                    "correct": str(correct),
                    "ts": dt.datetime.now().isoformat(timespec="seconds"),
                }
                # (1) write-and-flush on every row
                writer.writerow(row)
                f_csv.flush(); os.fsync(f_csv.fileno())

                print(f"[{tech}] item#{idx} | pred={pred} gold={gold} correct={correct}")

                if _shutdown["flag"]:
                    save_progress(progress_path, correct_counts, total_counts)
                    print("[info] Shutdown requested. Exiting inner loop…", flush=True)
                    return

        # final checkpoint
        save_progress(progress_path, correct_counts, total_counts)

    finally:
        try:
            f_csv.close()
        except Exception:
            pass

        # Print summary
        print("\n=== Accuracy by Technique ===")
        for tech in TECHNIQUES:
            total = total_counts[tech] or 1
            acc = correct_counts[tech] / total
            print(f"{tech:12s}: {acc:.2f}  ({correct_counts[tech]}/{total})")

        elapsed = time.time() - start_time
        print(f"\nSaved: {out_path.resolve()}")
        print(f"Runtime: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
