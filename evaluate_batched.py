"""
Batched local-model evaluation with resume and paper-table CSV export.

Three changes over the previous runner, each aimed at getting two valid full
runs out of a free GPU session:

1. BATCHING. Prompts are grouped and generated together, sorted by length so
   padding is minimal. On a T4 this is typically 5-8x faster than one call at a
   time, which is the difference between a run that finishes inside a session
   and one that does not.

2. RESUME. Every prediction is appended to a .jsonl as it is produced. If the
   session dies, rerunning the identical command skips completed work and
   continues. Nothing is lost to a disconnect.

3. CSV EXPORT. Writes CSVs whose columns match the paper's result tables, so
   numbers move from harness to manuscript without being retyped. Hand-copying
   is where transcription errors enter a paper.

Typical use (Kaggle, T4, internet on):

    !pip -q install -U transformers accelerate bitsandbytes
    !python evaluate_batched.py --suite suite.jsonl \
        --hf-model Qwen/Qwen2.5-7B-Instruct \
        --limit 300 --style cot --max-new-tokens 1400 \
        --batch-size 8 --tag cot300

    !python evaluate_batched.py --suite suite.jsonl \
        --hf-model Qwen/Qwen2.5-7B-Instruct \
        --limit 300 --style direct --max-new-tokens 300 \
        --batch-size 16 --tag direct300

Then merge both conditions into the final tables (no GPU needed):

    !python evaluate_batched.py --merge cot300 direct300 --suite suite.jsonl --limit 300

VALIDITY GATE: a run whose unparsable rate exceeds 2% is marked INVALID in the
CSV and must be repeated with a larger --max-new-tokens. Do not put an invalid
run into the paper: unparsable calls are scored as negative predictions, so they
bias judgment accuracy downward rather than at random.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
from typing import Dict, List, Tuple

import evaluate
from evaluate import (
    PositionPrior, boot_ci, macro_f1, mde, modal, parse_json, render_prompt,
)

Task = Tuple[str, str, int]  # (item_id, lang, run_index)


# --------------------------------------------------------------------------
# Batched generation
# --------------------------------------------------------------------------

class BatchedLocalModel:
    def __init__(self, model_id: str, four_bit: bool = True, greedy: bool = True,
                 temperature: float = 0.7, max_new_tokens: int = 1400,
                 seed: int = 0, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id
        self.greedy = greedy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_id)
        # decoder-only models must be LEFT-padded for batched generation,
        # otherwise short prompts continue from the middle of a pad run
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = {"device_map": "auto", "dtype": torch.float16}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()

    def _chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)

    def generate_batch(self, prompts: List[str], seed_base: int) -> List[str]:
        from transformers import set_seed
        torch = self.torch
        set_seed(seed_base)

        texts = [self._chat(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)

        gen = dict(max_new_tokens=self.max_new_tokens,
                   pad_token_id=self.tok.pad_token_id)
        if self.greedy:
            gen["do_sample"] = False
        else:
            gen.update(do_sample=True, temperature=self.temperature, top_p=0.95)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen)

        n_in = enc["input_ids"].shape[1]
        return [self.tok.decode(row[n_in:], skip_special_tokens=True) for row in out]


# --------------------------------------------------------------------------
# Task planning, resume, execution
# --------------------------------------------------------------------------

def plan_tasks(items, langs, runs) -> List[Task]:
    return [(it["id"], lang, r)
            for r in range(runs) for it in items for lang in langs]


def load_done(path: str) -> Dict[Task, dict]:
    done: Dict[Task, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # truncated final line after a hard kill
            done[(rec["id"], rec["lang"], rec["run"])] = rec["pred"]
    return done


def run_batched(model, items, langs, runs, raw_path, seed) -> Dict[Task, dict]:
    by_id = {it["id"]: it for it in items}
    tasks = plan_tasks(items, langs, runs)
    done = load_done(raw_path)
    todo = [t for t in tasks if t not in done]

    if done:
        print(f"  resuming: {len(done)} done, {len(todo)} remaining")
    if not todo:
        return done

    # sort by prompt length so each batch pads as little as possible
    prompts = {t: render_prompt(by_id[t[0]], t[1]) for t in todo}
    todo.sort(key=lambda t: len(prompts[t]))

    bs = model.batch_size
    total = len(todo)
    with open(raw_path, "a", encoding="utf-8") as sink:
        for start in range(0, total, bs):
            chunk = todo[start:start + bs]
            outs = model.generate_batch([prompts[t] for t in chunk],
                                        seed_base=seed * 1_000_003 + start)
            for t, raw in zip(chunk, outs):
                pred = parse_json(raw)
                done[t] = pred
                sink.write(json.dumps(
                    {"id": t[0], "lang": t[1], "run": t[2], "pred": pred},
                    ensure_ascii=False) + "\n")
            sink.flush()
            print(f"    {min(start + bs, total)}/{total}", flush=True)
    return done


def to_preds(done: Dict[Task, dict], items, langs, runs):
    out = collections.defaultdict(list)
    for it in items:
        for lang in langs:
            for r in range(runs):
                p = done.get((it["id"], lang, r))
                if p is not None:
                    out[(it["id"], lang)].append(p)
    return out


# --------------------------------------------------------------------------
# Paper tables
# --------------------------------------------------------------------------

def unparsable_pct(done: Dict[Task, dict]) -> float:
    if not done:
        return 0.0
    bad = sum(1 for p in done.values() if p.get("unparsable"))
    return 100.0 * bad / len(done)


def _scores(items, preds, lang):
    judg, step = [], []
    for it in items:
        p = preds.get((it["id"], lang))
        if not p:
            continue
        m, _ = modal(p)
        judg.append((it["is_hallucination"], m["is_hallucination"]))
        if it["is_hallucination"]:
            step.append(1.0 if m["hallucination_step"] == it["hallucination_step"] else 0.0)
    acc = [1.0 if g == q else 0.0 for g, q in judg]
    return judg, acc, step


def write_tables(conditions, items, langs, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "table3_by_condition.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "model", "lang", "n_items", "n_hallucinated",
                    "judgment_acc", "judg_ci_lo", "judg_ci_hi", "macro_f1",
                    "steploc_acc", "step_ci_lo", "step_ci_hi",
                    "unparsable_pct", "validity"])
        for cond, d in conditions.items():
            for lang in langs:
                judg, acc, step = _scores(items, d["preds"], lang)
                if not acc:
                    continue
                jci, sci = boot_ci(acc), boot_ci(step)
                up = d["unparsable"]
                w.writerow([cond, d["model"], lang, len(acc), len(step),
                            f"{statistics.mean(acc):.4f}", f"{jci[0]:.4f}", f"{jci[1]:.4f}",
                            f"{macro_f1(judg):.4f}",
                            f"{statistics.mean(step):.4f}" if step else "",
                            f"{sci[0]:.4f}", f"{sci[1]:.4f}",
                            f"{up:.2f}", "VALID" if up <= 2.0 else "INVALID (>2% unparsable)"])

    prior = PositionPrior(items, seed=99)
    with open(os.path.join(outdir, "table4_by_category.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "category", "n",
                    "steploc_ar", "ar_ci_lo", "ar_ci_hi",
                    "steploc_en", "en_ci_lo", "en_ci_hi", "position_matched_chance"])
        cats = sorted({it["category"] for it in items if it["is_hallucination"]})
        for cond, d in conditions.items():
            for cat in cats:
                sub = [it for it in items if it["category"] == cat]
                row = [cond, cat, len(sub)]
                for lang in langs:
                    vals = []
                    for it in sub:
                        p = d["preds"].get((it["id"], lang))
                        if not p:
                            continue
                        m, _ = modal(p)
                        vals.append(1.0 if m["hallucination_step"] == it["hallucination_step"] else 0.0)
                    ci = boot_ci(vals)
                    row += [f"{statistics.mean(vals):.4f}" if vals else "",
                            f"{ci[0]:.4f}" if vals else "", f"{ci[1]:.4f}" if vals else ""]
                chance = []
                for it in sub:
                    hits = [1.0 if prior.predict(it, langs[0])["hallucination_step"]
                            == it["hallucination_step"] else 0.0 for _ in range(400)]
                    chance.append(statistics.mean(hits))
                row.append(f"{statistics.mean(chance):.4f}" if chance else "")
                w.writerow(row)

    if len(langs) == 2:
        a, b = langs
        with open(os.path.join(outdir, "table5_cross_lingual.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["condition", "n_pairs", "paired_diff", "ci_lo", "ci_hi",
                        "discordant_pairs", "min_detectable_diff_80pct"])
            for cond, d in conditions.items():
                diffs, disc = [], 0
                for it in items:
                    if not it["is_hallucination"]:
                        continue
                    pa, pb = d["preds"].get((it["id"], a)), d["preds"].get((it["id"], b))
                    if not pa or not pb:
                        continue
                    ma, mb = modal(pa)[0], modal(pb)[0]
                    ca = 1.0 if ma["hallucination_step"] == it["hallucination_step"] else 0.0
                    cb = 1.0 if mb["hallucination_step"] == it["hallucination_step"] else 0.0
                    diffs.append(ca - cb)
                    disc += int(ca != cb)
                if not diffs:
                    continue
                ci = boot_ci(diffs)
                pdisc = disc / len(diffs)
                w.writerow([cond, len(diffs), f"{statistics.mean(diffs):.4f}",
                            f"{ci[0]:.4f}", f"{ci[1]:.4f}", disc,
                            f"{mde(len(diffs), max(pdisc, 0.05)):.4f}"])

    with open(os.path.join(outdir, "table6_dispersion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mean_modal_share", "min_modal_share", "non_unanimous", "n_cells"])
        for cond, d in conditions.items():
            shares = [modal(v)[1] for v in d["preds"].values() if v]
            if not shares:
                continue
            w.writerow([cond, f"{statistics.mean(shares):.4f}", f"{min(shares):.4f}",
                        sum(1 for s in shares if s < 1.0), len(shares)])

    print(f"\nwrote CSV tables to {outdir}/")
    for n in ("table3_by_condition", "table4_by_category",
              "table5_cross_lingual", "table6_dispersion"):
        print(f"  {n}.csv")


# --------------------------------------------------------------------------

def load_items(path, limit, offset):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.jsonl")
    ap.add_argument("--hf-model", default="")
    ap.add_argument("--tag", default="run", help="names the raw prediction file")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="tags to merge into final CSV tables; no GPU needed")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--langs", default="ar,en")
    ap.add_argument("--style", choices=("direct", "cot"), default="cot")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sampled", dest="greedy", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-4bit", dest="four_bit", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="tables")
    a = ap.parse_args()

    langs = a.langs.split(",")
    items = load_items(a.suite, a.limit, a.offset)

    if a.merge:
        conditions = {}
        for tag in a.merge:
            raw = f"raw_{tag}.jsonl"
            if not os.path.exists(raw):
                print(f"  ! missing {raw}, skipping")
                continue
            done = load_done(raw)
            runs = max((t[2] for t in done), default=0) + 1
            conditions[tag] = {
                "preds": to_preds(done, items, langs, runs),
                "unparsable": unparsable_pct(done),
                "model": "(from raw file)",
            }
            print(f"  {tag}: {len(done)} predictions, "
                  f"{conditions[tag]['unparsable']:.2f}% unparsable")
        if conditions:
            write_tables(conditions, items, langs, a.outdir)
        return

    if not a.hf_model:
        ap.error("--hf-model is required unless --merge is used")

    evaluate.STYLE = a.style
    max_new = a.max_new_tokens or (1400 if a.style == "cot" else 300)
    raw_path = f"raw_{a.tag}.jsonl"

    print(f"items {len(items)} (offset {a.offset})  langs {langs}  runs {a.runs}  "
          f"style {a.style}  max_new {max_new}  batch {a.batch_size}")
    print(f"raw predictions -> {raw_path}  (rerun the same command to resume)")

    model = BatchedLocalModel(a.hf_model, four_bit=a.four_bit, greedy=a.greedy,
                              temperature=a.temperature, max_new_tokens=max_new,
                              seed=a.seed, batch_size=a.batch_size)
    done = run_batched(model, items, langs, a.runs, raw_path, a.seed)

    up = unparsable_pct(done)
    print(f"\nunparsable: {up:.2f}%  -> "
          f"{'VALID' if up <= 2.0 else 'INVALID, raise --max-new-tokens and rerun'}")

    write_tables({a.tag: {"preds": to_preds(done, items, langs, a.runs),
                          "unparsable": up, "model": a.hf_model}},
                 items, langs, a.outdir)


if __name__ == "__main__":
    main()
