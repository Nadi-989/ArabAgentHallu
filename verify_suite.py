#!/usr/bin/env python3
"""verify_suite.py — deterministic, no-intervention audit of ArabAgentHallu.

Recomputes, from suite.jsonl and (when present) the raw prediction files,
every suite-level and subset-level number reported in the paper:

  * composition, length statistics, category/length balance        (Sec. 3.2)
  * requested design-grid census parsed from item ids (45 x 10)    (R1.6/R1.10)
  * requested->achieved position transition matrix                 (R1.6/R1.9)
  * Table 1 achieved position-by-category counts                   (Table 1)
  * analytic floors: uniform, first-step, modal position, prior    (Table 2)
  * keyword-floor behaviour on the rendered text (cue list below)  (Table 2, R2.4)
  * counterfactual label invariants (a)-(c) + final-step exclusion (Sec. 3.1)
  * structural AR/EN mirror                                        (Sec. 3.2)
  * subset composition, Table 3/5 statistics, per-condition
    realized MDD from the raw predictions                          (Tables 3,5; R1.7/R1.8)

Exit code 0 iff every check passes. Usage:
    python verify_suite.py [--suite suite.jsonl] [--preds results/*.jsonl]
"""
import argparse, glob, json, math, re, sys
from collections import Counter

CUES = ("بدل", "خطأ", "يناقض", "instead", "wrong", "contradict")  # keyword floor
ID_RE = re.compile(r"^h-([a-z-]+?)-(early|mid|late)-(short|medium|long)-\d{4}$")
CATS = ["planning", "retrieval", "reasoning", "human-interaction", "tool-use"]
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def pband(s, n):
    r = s / n
    return "early" if r <= 1 / 3 else ("mid" if r <= 2 / 3 else "late")


def lband(n):
    return "short" if n <= 9 else ("medium" if n <= 12 else "long")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.jsonl")
    ap.add_argument("--preds", nargs="*", default=None,
                    help="raw prediction files (default: results/raw_*.jsonl, raw_*.jsonl)")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.suite, encoding="utf-8")]
    H = [x for x in items if x["is_hallucination"]]
    C = [x for x in items if not x["is_hallucination"]]

    print("== Suite composition ==")
    check("900 items, 450/450", (len(items), len(H), len(C)) == (900, 450, 450))
    L = sorted(x["n_steps"] for x in items)
    med = (L[449] + L[450]) / 2
    check("lengths 7-15, median 11, mean 11.05",
          (L[0], L[-1]) == (7, 15) and med == 11 and abs(mean(L) - 11.05) < 0.005,
          f"min={L[0]} max={L[-1]} median={med} mean={mean(L):.2f}")
    per_cat = Counter(x["category"] for x in H)
    check("90 hallucinated per category", all(per_cat[c] == 90 for c in CATS))
    cl = Counter((x["category"], lband(x["n_steps"])) for x in H)
    check("category x length balance = 30 exactly",
          all(cl[(c, b)] == 30 for c in CATS for b in ("short", "medium", "long")))

    print("== Requested design grid (from item ids) ==")
    req = Counter()
    for x in H:
        m = ID_RE.match(x["id"])
        check(f"id parses: {x['id']}", bool(m)) if not m else req.update(
            [(m.group(1), m.group(2), m.group(3))])
    check("45 requested cells x depth 10", len(req) == 45 and set(req.values()) == {10})

    print("== Requested -> achieved position transition ==")
    trans = Counter()
    for x in H:
        m = ID_RE.match(x["id"])
        trans[(m.group(2), pband(x["hallucination_step"], x["n_steps"]))] += 1
    for r in ("early", "mid", "late"):
        row = {ach: trans.get((r, ach), 0) for ach in ("early", "mid", "late")}
        print(f"    requested {r:5s} -> {row}   (row sum {sum(row.values())})")
    check("transition rows each sum to 150",
          all(sum(trans.get((r, ac), 0) for ac in ("early", "mid", "late")) == 150
              for r in ("early", "mid", "late")))

    print("== Table 1 (achieved position by category) ==")
    t1 = {c: Counter() for c in CATS}
    for x in H:
        t1[x["category"]][pband(x["hallucination_step"], x["n_steps"])] += 1
    for c in CATS:
        print(f"    {c:18s} early={t1[c]['early']:3d} mid={t1[c]['mid']:3d} "
              f"late={t1[c]['late']:3d}")
    check("structural zeros: planning-late=0, tool-use-early=0",
          t1["planning"]["late"] == 0 and t1["tool-use"]["early"] == 0)

    print("== Analytic floors (Table 2) ==")
    steps = [x["hallucination_step"] for x in H]
    uni = mean(1 / x["n_steps"] for x in H)
    first = sum(1 for s in steps if s == 1) / len(H)
    cnt = Counter(steps)
    modal_step, modal_n = cnt.most_common(1)[0]
    prior = sum((v / len(H)) ** 2 for v in cnt.values())
    check("uniform-random = 0.097", round(uni, 3) == 0.097, f"{uni:.4f}")
    check("always-first = 0.084", round(first, 3) == 0.084, f"{first:.4f}")
    check("modal-position = 0.164", round(modal_n / len(H), 3) == 0.164,
          f"step {modal_step}: {modal_n}/450")
    print(f"    position-prior expected step-loc (full suite): {prior:.3f}")

    print("== Keyword floor on rendered text ==")
    def blob(x, lang):
        T = x["languages"][lang]
        return T["query"] + " " + " ".join(
            s["thought"] + " " + s["action"] + " " + s["observation"] for s in T["steps"])
    fires = sum(any(c in blob(x, lg) for c in CUES) for x in items for lg in ("ar", "en"))
    check("keyword floor never fires (all-negative, judgment 0.500)", fires == 0,
          f"fires={fires}")

    print("== Counterfactual label invariants ==")
    check("(b,c) injection is earliest member of H(tau)",
          all(x["hallucination_step"] == min(x["responsible_steps"]) for x in H))
    check("final answer-emitting step excluded from H(tau)",
          all(x["n_steps"] not in x["responsible_steps"] for x in H))
    check("(a) every hallucinated item flips the answer",
          all(x["produced_answer"]["ar"] != x["gold_answer"]["ar"] for x in H))
    check("every clean item matches gold",
          all(x["produced_answer"]["ar"] == x["gold_answer"]["ar"] for x in C))

    print("== Structural AR/EN mirror ==")
    check("step counts identical across renderings",
          all(len(x["languages"]["ar"]["steps"]) == len(x["languages"]["en"]["steps"])
              == x["n_steps"] for x in items))

    # ---- raw predictions, if present ----
    pred_files = a.preds if a.preds is not None else sorted(
        set(glob.glob("results/raw_*.jsonl") + glob.glob("raw_*.jsonl")))
    by_id = {x["id"]: x for x in items}
    for pf in pred_files:
        print(f"== Raw predictions: {pf} ==")
        rows = [json.loads(l) for l in open(pf, encoding="utf-8")]
        ids = sorted({r["id"] for r in rows})
        hs = [i for i in ids if i.startswith("h-")]
        print(f"    rows={len(rows)}  items={len(ids)}  "
              f"hallucinated={len(hs)}  clean={len(ids) - len(hs)}")
        for lang in ("ar", "en"):
            P = {r["id"]: r["pred"] for r in rows if r["lang"] == lang}
            jac = mean(
                [1 if bool(P[i]["is_hallucination"]) == by_id[i]["is_hallucination"]
                 else 0 for i in ids])
            # Eq. (6) of the paper: step-localization scores the step field
            # alone on hallucinated items, independent of the verdict field
            # (judgment and localization are reported as separate capabilities).
            loc = mean([1 if P[i]["hallucination_step"]
                        == by_id[i]["hallucination_step"] else 0 for i in hs])
            strict = mean([1 if P[i]["is_hallucination"]
                           and P[i]["hallucination_step"]
                           == by_id[i]["hallucination_step"] else 0 for i in hs])
            print(f"    {lang}: judgment_acc={jac:.4f}  "
                  f"steploc_acc(Eq.6)={loc:.4f}  "
                  f"[verdict-conditioned diagnostic: {strict:.4f}]")
        # paired cross-lingual stats on hallucinated pairs
        Par = {r["id"]: r["pred"] for r in rows if r["lang"] == "ar"}
        Pen = {r["id"]: r["pred"] for r in rows if r["lang"] == "en"}
        d = []
        for i in hs:
            ca = 1 if Par[i]["hallucination_step"] == by_id[i]["hallucination_step"] else 0
            ce = 1 if Pen[i]["hallucination_step"] == by_id[i]["hallucination_step"] else 0
            d.append(ca - ce)
        n = len(d)
        md = mean(d)
        sd = math.sqrt(mean([(x - md) ** 2 for x in d]))
        disc = sum(1 for x in d if x != 0)
        mdd = 2.8 * sd / math.sqrt(n) if n else float("nan")
        print(f"    paired diff={md:+.4f}  discordant={disc}/{n}  "
              f"sigma_d={sd:.4f}  realized MDD(80%)={mdd:.4f}")

    print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
