#!/usr/bin/env python3
"""make_manifest.py — machine-readable design-grid manifest (reviewer R1.10).

Writes manifest.csv with one row per suite item:
  id, is_hallucination, category, skeleton, domain, n_steps, length_band,
  requested_position_band (parsed from the item id; the sampler fills the
  requested 45-cell grid to equal depth), achieved_position_band (position
  band of the counterfactually verified responsible step t*), t_star,
  H_size (|H(tau)| after final-step exclusion), n_fillers.

All generated items in the released suite are accepted items: candidates
failing the counterfactual admission conditions (a)-(c) are rejected and
resampled inside the generator and never emitted (see core.make_item).

Usage:  python make_manifest.py [--suite suite.jsonl] [--out manifest.csv]
"""
import argparse, csv, json, re

ID_RE = re.compile(r"^h-([a-z-]+?)-(early|mid|late)-(short|medium|long)-\d{4}$")


def pband(s, n):
    r = s / n
    return "early" if r <= 1 / 3 else ("mid" if r <= 2 / 3 else "late")


def lband(n):
    return "short" if n <= 9 else ("medium" if n <= 12 else "long")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite.jsonl")
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "is_hallucination", "category", "skeleton", "domain",
                    "n_steps", "length_band", "requested_position_band",
                    "achieved_position_band", "t_star", "H_size", "n_fillers"])
        n = 0
        for line in open(a.suite, encoding="utf-8"):
            x = json.loads(line)
            m = ID_RE.match(x["id"]) if x["is_hallucination"] else None
            w.writerow([
                x["id"], int(x["is_hallucination"]), x["category"] or "",
                x["skeleton"], x["domain"], x["n_steps"], lband(x["n_steps"]),
                m.group(2) if m else "",
                pband(x["hallucination_step"], x["n_steps"])
                if x["is_hallucination"] else "",
                x["hallucination_step"] or "",
                len(x["responsible_steps"]) if x["responsible_steps"] else "",
                len(x["filler_steps"]),
            ])
            n += 1
    print(f"wrote {a.out}: {n} rows")


if __name__ == "__main__":
    main()
