# ArabAgentHallu

**A bilingual Arabic–English benchmark for hallucination judgment and step attribution in LLM-based agents.**

900 structurally mirrored agent trajectories (450 hallucinated / 450 clean, 7–15 steps) across five hallucination categories, with counterfactually verified ground truth: every responsible step is derived by execution (restore-and-rerun), not by annotation. Arabic and English surfaces are rendered from the same underlying program, isolating the language variable for paired cross-lingual statistics.

## Repository layout (flat)

```
├── core.py                  # trajectory programs, injection machinery
├── domains.py               # five domain lexicons
├── generate.py              # suite generation (deterministic, seeded)
├── build.py                 # design grid: 5 categories × 3 position bands × 3 length bands
├── evaluate.py              # reference evaluation protocol (v1.1: scripted §3.3.3 scaffolds + last-JSON parser)
├── evaluate_batched.py      # batched GPU runner (resume-capable) + table builder (v1.1: adds freecot style)
├── evaluate_local.py        # local single-item runner
├── suite.jsonl              # the released 900-item suite (frozen artefact)
├── manifest.csv             # one row per item: requested/achieved bands, t*, |H(τ)|, fillers
├── make_manifest.py         # regenerates manifest.csv deterministically from suite.jsonl
├── verify_suite.py          # deterministic audit: re-derives every reported number → ALL CHECKS PASSED
│
│   # raw predictions (300-item subset, 600 rows = 300 items × 2 languages unless noted)
├── raw_qwen7b_cot300.jsonl      # Qwen2.5-7B, structured CoT  (paper Tables 3–5)
├── raw_qwen7b_free300.jsonl     # Qwen2.5-7B, free CoT        (paper §4.3)
├── raw_rep60x3.jsonl            # Qwen2.5-7B, k=3 repetitions × 60 items × 2 languages (paper §4.6)
├── raw_llama8b_cot300.jsonl     # Llama-3.1-8B, structured CoT   (paper §4.9, Table 7)
├── raw_llama8b_direct300.jsonl  # Llama-3.1-8B, direct           (paper §4.9, Table 7)
├── raw_arab_cot300.jsonl        # SILMA-9B, structured CoT       (paper §4.9, Table 7)
├── raw_arab_direct300.jsonl     # SILMA-9B, direct               (paper §4.9, Table 7)
│
├── table3_by_condition.csv  # Tables 3–6 as CSV (bootstrap CIs)
├── table4_by_category.csv
├── table5_cross_lingual.csv
├── table6_dispersion.csv
└── table7_multimodel.csv    # cross-model comparison with Wilson CIs and validity flags
```

## Reproducing the suite

```bash
python generate.py --seed 13 --out suite.jsonl
```

Generation is deterministic: seed **13** reproduces the released `suite.jsonl` byte-identically (verified on three independent machines with different Python versions).

## Verifying every reported number

```bash
python verify_suite.py          # suite census, floors, invariants, transition matrix,
                                # and per-file statistics for every raw_*.jsonl present
python make_manifest.py         # regenerates manifest.csv
```

`verify_suite.py` terminates with `ALL CHECKS PASSED` and re-derives, with no manual
intervention: the 45×10 requested-cell census (from the item identifiers), the
requested→achieved position transition matrix, all Table 1 counts, the analytic floors,
the keyword-floor behaviour, the counterfactual invariants (a)–(c) with final-step
exclusion, the AR/EN structural mirror, and all judgment / step-localization / paired
cross-lingual statistics from the raw prediction files.

## Reproducing an evaluation run

```bash
python evaluate_batched.py \
    --suite suite.jsonl \
    --hf-model "Qwen/Qwen2.5-7B-Instruct" \
    --greedy --seed 0 \
    --limit 300 --style cot \
    --batch-size 4 --tag qwen7b_cot300
# then build the tables:
python evaluate_batched.py --suite suite.jsonl --merge qwen7b_cot300
```

Styles: `cot` (structured scaffold), `freecot` (unconstrained reasoning), `direct`
(JSON-first). Interrupted runs resume automatically from the raw prediction file.

### Harness versions

* **v1.1** (current files): the §3.3.3 prompt scaffolds are appended programmatically to
  every prompt, the parser extracts the last JSON object in the reply, and the `freecot`
  style is available. All runs added in the revision (`raw_qwen7b_free300`, `raw_rep60x3`,
  `raw_llama8b_*`, `raw_arab_*`) used v1.1.
* **v1.0** (git tag `v1.0`): the original release used for `raw_qwen7b_cot300.jsonl`; it
  shared the decoding budgets and parser contract but did not append the literal scaffold
  text. The raw prediction file of the v1.0 *direct* condition was not preserved; the
  direct-condition numbers in the paper are those of the original run log, and the
  condition is exactly reproducible with the command above (`--style direct`).

## Headline results (300-item subset, 4-bit NF4, greedy)

| | Arabic | English |
|---|---|---|
| Qwen2.5-7B judgment | 0.607 [0.550–0.660] | 0.627 [0.571–0.680] |
| Qwen2.5-7B step-localization | 0.396 [0.320–0.477] | 0.438 [0.359–0.519] |

Paired Arabic−English step-localization difference: **−0.042** for Qwen2.5-7B — but the
gap is strongly **model-dependent**: Llama-3.1-8B shows **−0.472** under the direct
protocol (0.076 AR vs 0.549 EN; 80/144 discordant pairs), and output-format compliance
itself degrades in Arabic (Llama structured-CoT unparsable 34.7% AR vs 7.3% EN; SILMA-9B
100% both). See `table7_multimodel.csv` and §4.9 of the paper.

## License

Code: MIT. Suite data (`suite.jsonl`) and results: CC BY 4.0.

## Citation

See `CITATION.cff`.
