"""
Generate a balanced, validated ArabAgentHallu suite.

    python generate.py --n-hallucinated 400 --n-clean 400 --out suite.jsonl

The sampler enumerates the full design grid

    category (5) x position bucket (3) x length band (3)

and fills every cell to equal depth, so per-cell support is n/45 rather than 1,
and no category is confounded with position or length. Every emitted item has
passed the counterfactual checks in core.make_item.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from typing import Any, Dict, List

from build import BUCKETS, CATEGORIES, bucket_of, sample_item
from core import BuildError

LENGTH_BANDS = {"short": (0, 2), "medium": (3, 5), "long": (6, 8)}  # filler counts


def build_suite(
    n_hallucinated: int, n_clean: int, seed: int, max_tries: int = 60
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    grid = [(c, b, band) for c in CATEGORIES for b in BUCKETS for band in LENGTH_BANDS]
    per_cell = max(1, n_hallucinated // len(grid))

    for cat, bucket, band in grid:
        lo, hi = LENGTH_BANDS[band]
        made = 0
        tries = 0
        while made < per_cell and tries < per_cell * max_tries:
            tries += 1
            nf = rng.randint(lo, hi)
            iid = f"h-{cat}-{bucket}-{band}-{made:04d}"
            try:
                items.append(sample_item(rng, cat, bucket, nf, iid))
                made += 1
            except BuildError:
                continue

    made = 0
    tries = 0
    while made < n_clean and tries < n_clean * max_tries:
        tries += 1
        band = rng.choice(list(LENGTH_BANDS))
        lo, hi = LENGTH_BANDS[band]
        try:
            items.append(sample_item(rng, None, rng.choice(BUCKETS),
                                     rng.randint(lo, hi), f"c-{made:04d}"))
            made += 1
        except BuildError:
            continue

    rng.shuffle(items)
    return items


def report(items: List[Dict[str, Any]]) -> None:
    hal = [i for i in items if i["is_hallucination"]]
    lens = [i["n_steps"] for i in items]
    print(f"\nitems: {len(items)}  hallucinated: {len(hal)}  clean: {len(items) - len(hal)}")
    print(f"steps: min {min(lens)}  median {statistics.median(lens)}  max {max(lens)}  "
          f"mean {statistics.mean(lens):.1f}")

    print("\nper category x position bucket (hallucinated):")
    tab = collections.Counter((i["category"], i["position_bucket"]) for i in hal)
    print(f"  {'':<20}" + "".join(f"{b:>8}" for b in BUCKETS))
    for c in CATEGORIES:
        print(f"  {c:<20}" + "".join(f"{tab[(c, b)]:>8}" for b in BUCKETS))

    print("\ndomains: " + ", ".join(
        f"{k}={v}" for k, v in sorted(collections.Counter(i["domain"] for i in items).items())))

    # Analytic random baseline: expected step-localization accuracy of a model
    # that judges perfectly but guesses the step uniformly at random.
    exp = statistics.mean(1 / i["n_steps"] for i in hal)
    first = sum(1 for i in hal if i["hallucination_step"] == 1) / len(hal)
    last = sum(1 for i in hal if i["hallucination_step"] == i["n_steps"]) / len(hal)
    print(f"\nreference floors on this suite:")
    print(f"  uniform-random step guess : {exp:.3f}")
    print(f"  always-blame-first-step   : {first:.3f}")
    print(f"  always-blame-last-step    : {last:.3f}")

    mode = collections.Counter(i["hallucination_step"] for i in hal).most_common(1)[0]
    print(f"  always-blame-step-{mode[0]:<9}: {mode[1] / len(hal):.3f}   "
          f"<- the number any reported accuracy must beat")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hallucinated", type=int, default=450)
    ap.add_argument("--n-clean", type=int, default=450)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="suite.jsonl")
    a = ap.parse_args()

    items = build_suite(a.n_hallucinated, a.n_clean, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    report(items)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
