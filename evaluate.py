"""
Evaluation harness.

Three things this does that a single-pass script cannot:

1. Repeats every item n times and reports dispersion. The pilot already observed
   run-to-run variation at temperature 0; until that variation is quantified,
   any single-run table is one draw from an unmeasured distribution.

2. Scores against a *position-matched* random baseline, per category. Injection
   position is structurally correlated with category — planning errors cannot be
   late, tool errors cannot be early, in this suite and in real agents alike. A
   model with nothing but a positional prior therefore scores well on some
   categories. Reporting raw per-category accuracy without this baseline
   attributes to comprehension what may be positional luck.

3. Treats the Arabic/English comparison as paired, because each record is one
   program rendered twice. A paired bootstrap on the per-item difference is far
   more sensitive than comparing two independent means, and it yields the
   interval that an "absence of a cross-lingual gap" claim actually requires.

Usage
-----
    python evaluate.py --suite suite.jsonl --dry-run --runs 5
    python evaluate.py --suite suite.jsonl --model MODEL --base-url URL \
        --api-key-env KEY --runs 5 --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------

PROMPT = {
    "ar": """أنت تحلّل مسار عمل وكيل ذكاء اصطناعي.

الطلب:
{query}

المسار:
{steps}

الجواب الذي أنتجه الوكيل: {answer}

حدّد: هل يحتوي المسار على هلوسة أدّت إلى جواب خاطئ؟ وإن كان كذلك، ما رقم الخطوة
المسؤولة؟ الخطوة المسؤولة هي أبكر خطوة لو صُحّحت وحدها لأصبح الجواب صحيحاً — أي
الخطوة التي دخل فيها الخطأ، لا الخطوة التي ظهر فيها أثره.

أعد كائن JSON فقط، بلا أي نص آخر:
{{"is_hallucination": true/false, "hallucination_step": رقم أو null, "hallucination_reason": "سبب موجز"}}""",
    "en": """You are analysing the trajectory of an AI agent.

Query:
{query}

Trajectory:
{steps}

Answer produced by the agent: {answer}

Decide: does the trajectory contain a hallucination that made the answer wrong?
If so, which step number is responsible? The responsible step is the earliest
step whose correction alone would make the answer correct — the step where the
error enters, not the step where its effect becomes visible.

Return only a JSON object, with no other text:
{{"is_hallucination": true/false, "hallucination_step": number or null, "hallucination_reason": "brief reason"}}""",
}


def render_prompt(item: Dict[str, Any], lang: str) -> str:
    side = item["languages"][lang]
    steps = "\n".join(
        f"[{s['step']}] thought: {s['thought']} | action: {s['action']} | observation: {s['observation']}"
        for s in side["steps"]
    )
    return PROMPT[lang].format(
        query=side["query"], steps=steps, answer=item["produced_answer"][lang]
    )


# --------------------------------------------------------------------------
# Prediction sources
# --------------------------------------------------------------------------

class Predictor:
    name = "base"

    def predict(self, item: Dict[str, Any], lang: str) -> Dict[str, Any]:
        raise NotImplementedError


class KeywordFloor(Predictor):
    """Deterministic floor: flags on contradiction cues, always blames step 1."""

    name = "keyword-floor"
    CUES = ("بدل", "خطأ", "يناقض", "instead", "wrong", "contradict")

    def predict(self, item, lang):
        blob = " ".join(
            s["observation"] + s["thought"] for s in item["languages"][lang]["steps"]
        )
        hit = any(c in blob for c in self.CUES)
        return {"is_hallucination": hit, "hallucination_step": 1 if hit else None}


class UniformRandom(Predictor):
    """Judges perfectly, guesses the step uniformly. The honest chance level."""

    name = "uniform-random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def predict(self, item, lang):
        h = item["is_hallucination"]
        return {
            "is_hallucination": h,
            "hallucination_step": self.rng.randint(1, item["n_steps"]) if h else None,
        }


class PositionPrior(Predictor):
    """
    Judges perfectly, guesses the step from the *empirical* position distribution
    of the suite. This is the baseline that a category-wise score must beat before
    it can be called attribution rather than a positional habit.
    """

    name = "position-prior"

    def __init__(self, items: Sequence[Dict[str, Any]], seed=0):
        self.rng = random.Random(seed)
        hal = [i for i in items if i["is_hallucination"]]
        self.frac = [i["hallucination_step"] / i["n_steps"] for i in hal] or [0.5]

    def predict(self, item, lang):
        h = item["is_hallucination"]
        if not h:
            return {"is_hallucination": False, "hallucination_step": None}
        f = self.rng.choice(self.frac)
        step = min(item["n_steps"], max(1, round(f * item["n_steps"])))
        return {"is_hallucination": True, "hallucination_step": step}


class StubModel(Predictor):
    """
    Offline stand-in used by --dry-run: judges well, localises with an early-step
    bias, and systematically mislocates human-interaction items to the endorsement
    step — the pilot's documented failure mode. It exists so the harness, the
    dispersion statistics and the baselines can be exercised without a live API.
    """

    name = "stub-model"

    def __init__(self, seed=0, noise=0.15):
        self.rng = random.Random(seed)
        self.noise = noise

    def predict(self, item, lang):
        h = item["is_hallucination"]
        if self.rng.random() < 0.05:
            h = not h
        if not h:
            return {"is_hallucination": False, "hallucination_step": None}
        t = item["hallucination_step"] or 1
        if item["category"] == "human-interaction":
            t = min(item["n_steps"], t + 1)          # endorsement, not origin
        elif self.rng.random() < self.noise:
            t = min(item["n_steps"], max(1, t + self.rng.choice([-1, 1])))
        return {"is_hallucination": True, "hallucination_step": t}


class APIModel(Predictor):
    """OpenAI-compatible chat completions client, stdlib only."""

    def __init__(self, model, base_url, api_key, temperature=0.0,
                 delay=1.0, retries=6, timeout=120):
        self.name = model
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.temperature, self.delay, self.retries, self.timeout = (
            temperature, delay, retries, timeout)

    def predict(self, item, lang):
        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": render_prompt(item, lang)}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = json.loads(r.read())["choices"][0]["message"]["content"]
                time.sleep(self.delay)
                return parse_json(raw)
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError):
                time.sleep(self.delay * (2 ** attempt))
        return {"is_hallucination": False, "hallucination_step": None, "unparsable": True}


def parse_json(raw: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {"is_hallucination": False, "hallucination_step": None, "unparsable": True}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"is_hallucination": False, "hallucination_step": None, "unparsable": True}
    step = d.get("hallucination_step")
    try:
        step = int(step) if step is not None else None
    except (TypeError, ValueError):
        step = None
    return {"is_hallucination": bool(d.get("is_hallucination")),
            "hallucination_step": step,
            "reason": d.get("hallucination_reason", "")}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def macro_f1(pairs: Sequence[Tuple[bool, bool]]) -> float:
    def f1(pos: bool) -> float:
        tp = sum(1 for g, p in pairs if g == pos and p == pos)
        fp = sum(1 for g, p in pairs if g != pos and p == pos)
        fn = sum(1 for g, p in pairs if g == pos and p != pos)
        if tp == 0:
            return 0.0
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        return 2 * pr * rc / (pr + rc)
    return 0.5 * (f1(True) + f1(False))


def boot_ci(vals: Sequence[float], iters=2000, seed=7, alpha=0.05) -> Tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(iters):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters)]


def fmt(v: float, ci: Tuple[float, float]) -> str:
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def mde(n: int, p_discordant: float = 0.2) -> float:
    """
    Minimum detectable paired difference at 80% power, alpha .05 — the number the
    paper needs in order to say what 'no cross-lingual gap' rules out.
    """
    if n <= 0:
        return float("nan")
    return (1.96 + 0.84) * (p_discordant ** 0.5) / (n ** 0.5)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def run_predictor(pred: Predictor, items, langs, runs: int) -> Dict[Tuple[str, str], List[dict]]:
    out: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
    total = len(items) * len(langs) * runs
    done = 0
    for r in range(runs):
        for it in items:
            for lang in langs:
                out[(it["id"], lang)].append(pred.predict(it, lang))
                done += 1
                if done % 500 == 0:
                    print(f"    {done}/{total}", flush=True)
    return out


def modal(preds: List[dict]) -> Tuple[dict, float]:
    """Modal prediction plus the fraction of runs agreeing with it."""
    keys = [(p["is_hallucination"], p["hallucination_step"]) for p in preds]
    (k, c) = collections.Counter(keys).most_common(1)[0]
    return {"is_hallucination": k[0], "hallucination_step": k[1]}, c / len(keys)


def summarise(name, items, preds, langs, all_items) -> None:
    print(f"\n=== {name} ===")

    agree = [modal(preds[(i["id"], l)])[1] for i in items for l in langs]
    unparsed = sum(1 for i in items for l in langs
                   for p in preds[(i["id"], l)] if p.get("unparsable"))
    print(f"run-to-run agreement (modal share): mean {statistics.mean(agree):.3f}  "
          f"min {min(agree):.3f}  items never unanimous: "
          f"{sum(1 for a in agree if a < 1.0)}/{len(agree)}   unparsable calls: {unparsed}")

    for lang in langs:
        j, s = [], []
        for it in items:
            m, _ = modal(preds[(it["id"], lang)])
            j.append((it["is_hallucination"], m["is_hallucination"]))
            if it["is_hallucination"]:
                s.append(1.0 if m["hallucination_step"] == it["hallucination_step"] else 0.0)
        acc = [1.0 if g == p else 0.0 for g, p in j]
        print(f"  [{lang}] judgment acc {fmt(statistics.mean(acc), boot_ci(acc))}   "
              f"macro-F1 {macro_f1(j):.3f}   step-loc {fmt(statistics.mean(s), boot_ci(s))}  (n={len(s)})")

    # position-matched chance level, per category
    print("  per category — model vs position-matched chance:")
    pos_prior = PositionPrior(all_items, seed=99)
    for cat in sorted({i["category"] for i in items if i["is_hallucination"]}):
        sub = [i for i in items if i["category"] == cat]
        got, chance = [], []
        for it in sub:
            for lang in langs:
                m, _ = modal(preds[(it["id"], lang)])
                got.append(1.0 if m["hallucination_step"] == it["hallucination_step"] else 0.0)
                chance.append(statistics.mean(
                    1.0 if pos_prior.predict(it, lang)["hallucination_step"]
                    == it["hallucination_step"] else 0.0 for _ in range(200)))
        print(f"    {cat:<20} {fmt(statistics.mean(got), boot_ci(got)):<26}"
              f" chance {statistics.mean(chance):.3f}  (n={len(sub)})")

    # paired cross-lingual difference
    if len(langs) == 2:
        a, b = langs
        diffs, disc = [], 0
        for it in items:
            if not it["is_hallucination"]:
                continue
            ma, _ = modal(preds[(it["id"], a)])
            mb, _ = modal(preds[(it["id"], b)])
            ca = 1.0 if ma["hallucination_step"] == it["hallucination_step"] else 0.0
            cb = 1.0 if mb["hallucination_step"] == it["hallucination_step"] else 0.0
            diffs.append(ca - cb)
            disc += int(ca != cb)
        n = len(diffs)
        ci = boot_ci(diffs)
        p_disc = disc / n if n else 0.0
        print(f"  paired {a}-{b} step-loc difference: {fmt(statistics.mean(diffs), ci)}   "
              f"discordant pairs {disc}/{n}")
        print(f"  minimum detectable difference at 80% power: "
              f"{mde(n, max(p_disc, 0.05)):.3f}  <- claims of parity are bounded by this, "
              f"not by the point estimate")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.jsonl")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--langs", default="ar,en")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--out", default="predictions.json")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.suite, encoding="utf-8")]
    if a.limit:
        items = items[: a.limit]
    langs = a.langs.split(",")
    print(f"loaded {len(items)} items, langs={langs}, runs={a.runs}")

    preds_all = {}
    systems: List[Tuple[Predictor, int]] = [
        (KeywordFloor(), 1),
        (UniformRandom(seed=1), a.runs),
        (PositionPrior(items, seed=2), a.runs),
    ]
    if a.dry_run or not a.model:
        systems.append((StubModel(seed=3), a.runs))
    else:
        systems.append((APIModel(a.model, a.base_url, os.environ.get(a.api_key_env, ""),
                                 delay=a.delay), a.runs))

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

# === PKGC_STYLE_PATCH (package C): styles per paper §3.3.3 + last-JSON parser ===
STYLE = 'cot'
SCAFFOLD = {'cot': {'ar': '\nقبل إصدار الحكم اتبع الخطوات التالية بترتيبها: (1) استخرج من الطلب كل قيمة مرجعية منصوص عليها (أرقام، تعرفات، قنوات). (2) امش عبر خطوات المسار واحدة واحدة وقارن كل قيمة مستخدمة أو ناتجة بالمرجعيات المستخرجة. (3) بعد اكتمال المقارنة فقط، أصدر الحكم. اكتب هذا التحليل نصاً، ثم أنهِ ردك بكائن JSON واحد في سطره الأخير.', 'en': '\nBefore judging, follow these steps in order: (1) extract every reference value stated in the query (numbers, tariffs, channels); (2) walk the trajectory step by step and compare each used or produced value against the extracted references; (3) only after the comparison is complete, decide. Write this analysis as text, then END your reply with a single JSON object on the final line.'}, 'freecot': {'ar': '\nفكر بحرية كما تشاء قبل الحكم، ثم أنهِ ردك بكائن JSON واحد في سطره الأخير.', 'en': '\nReason freely in any way you like before judging, then END your reply with a single JSON object on the final line.'}, 'direct': {'ar': '\nأعد كائن JSON فوراً كأول شيء في ردك، دون أي تحليل أو نص قبله.', 'en': '\nReturn the JSON object immediately as the very first thing in your reply, with no analysis or text before it.'}}
_orig_render_prompt = render_prompt
def render_prompt(item, lang):
    return _orig_render_prompt(item, lang) + SCAFFOLD.get(STYLE, {}).get(lang, '')
_orig_parse_json = parse_json
def parse_json(raw):
    cands = re.findall(r'\{[^{}]*\}', raw or '', re.S)
    for c in reversed(cands):
        try:
            d = json.loads(c)
            if 'is_hallucination' in d:
                step = d.get('hallucination_step')
                try: step = int(step) if step is not None else None
                except (TypeError, ValueError): step = None
                return {'is_hallucination': bool(d.get('is_hallucination')),
                        'hallucination_step': step,
                        'reason': d.get('hallucination_reason', '')}
        except Exception:
            continue
    return _orig_parse_json(raw)
