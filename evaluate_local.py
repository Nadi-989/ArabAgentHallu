"""
Local-model evaluation — no API key, no quota, no provider nondeterminism.

Runs an open instruction-tuned model on the free Colab GPU and reuses the
harness in evaluate.py unchanged, so the metrics, baselines, dispersion and
paired cross-lingual statistics are identical to the API path.

Colab setup (T4 is enough for a 7B in 4-bit):
    !pip -q install transformers accelerate bitsandbytes
    Runtime > Change runtime type > T4 GPU

    !python evaluate_local.py --suite suite.jsonl \
        --hf-model Qwen/Qwen2.5-7B-Instruct --runs 3 --limit 150

Why this is better than a free API tier for your paper
------------------------------------------------------
Your pilot documented predictions changing between runs at temperature 0 and
attributed it to provider-side nondeterminism. Locally you control the seed, so
you can separate two things that were confounded: variation caused by sampling,
and variation caused by the provider. Run once with --greedy (seeded, temp 0)
and once with --temperature 0.7 across runs. If greedy decoding is stable and
sampled decoding is not, that is a clean, publishable finding rather than a
caveat.
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluate import (
    KeywordFloor, PositionPrior, Predictor, UniformRandom,
    parse_json, render_prompt, run_predictor, summarise,
)


class LocalModel(Predictor):
    def __init__(self, model_id: str, four_bit: bool = True,
                 greedy: bool = True, temperature: float = 0.7,
                 max_new_tokens: int = 160, seed: int = 0):
        self.name = model_id
        self.greedy = greedy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._call = 0

        self.tok = AutoTokenizer.from_pretrained(model_id)
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

    @torch.inference_mode()
    def predict(self, item, lang):
        # A fresh seed per call, derived deterministically: reruns of the same
        # item differ (so dispersion is measurable) but the whole experiment
        # reproduces exactly from --seed.
        self._call += 1
        set_seed(self.seed * 1_000_003 + self._call)

        msgs = [{"role": "user", "content": render_prompt(item, lang)}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)

        gen = dict(max_new_tokens=self.max_new_tokens,
                   pad_token_id=self.tok.eos_token_id)
        if self.greedy:
            gen["do_sample"] = False
        else:
            gen.update(do_sample=True, temperature=self.temperature, top_p=0.95)

        out = self.model.generate(**enc, **gen)
        raw = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        return parse_json(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.jsonl")
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--langs", default="ar,en")
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sampled", dest="greedy", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-4bit", dest="four_bit", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", default="predictions_local.json")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.suite, encoding="utf-8")]
    if a.limit:
        items = items[: a.limit]
    langs = a.langs.split(",")
    print(f"loaded {len(items)} items, langs={langs}, runs={a.runs}, "
          f"decoding={'greedy' if a.greedy else f'sampled T={a.temperature}'}")

    preds_all = {}
    systems = []
    if not a.skip_baselines:
        systems += [(KeywordFloor(), 1),
                    (UniformRandom(seed=1), a.runs),
                    (PositionPrior(items, seed=2), a.runs)]
    systems.append((LocalModel(a.hf_model, four_bit=a.four_bit, greedy=a.greedy,
                               temperature=a.temperature, seed=a.seed), a.runs))

    for pred, runs in systems:
        print(f"\nrunning {pred.name} ({runs} run(s))...")
        preds = run_predictor(pred, items, langs, runs)
        summarise(pred.name, items, preds, langs, items)
        preds_all[pred.name] = {f"{k[0]}|{k[1]}": v for k, v in preds.items()}

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(preds_all, f, ensure_ascii=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
