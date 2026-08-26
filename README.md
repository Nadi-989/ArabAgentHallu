# ArabAgentHallu

**A bilingual Arabic–English benchmark for hallucination judgment and step attribution in LLM-based agents.**

900 structurally mirrored agent trajectories (450 hallucinated / 450 clean, 7–15 steps) across five hallucination categories, with counterfactually verified ground truth: every responsible step is derived by execution (restore-and-rerun), not by annotation. Arabic and English surfaces are rendered from the same underlying program, isolating the language variable for paired cross-lingual statistics.

## Repository layout

```
├── generator/
│   ├── core.py          # trajectory programs, injection machinery
│   ├── domains.py       # four domain lexicons
│   ├── generate.py      # suite generation (deterministic, seeded)
│   └── build.py         # design grid: 5 categories x 3 position bands x 3 length bands
├── harness/
│   ├── evaluate.py           # reference evaluation protocol
│   ├── evaluate_local.py     # local single-item runner
│   └── evaluate_batched.py   # batched GPU runner (resume-capable) + table builder
├── suite.jsonl          # the released 900-item suite (frozen artefact)
├── results/
│   ├── raw_cot300.jsonl        # step-by-step condition, 300-item subset, 600 predictions
│   └── tables/                 # Tables 3-6 as CSV (bootstrap CIs)
└── paper/
    └── ArabAgentHallu_paper_v0_7.pdf
```

## Reproducing the suite

```bash
python generator/generate.py --seed 20250823 --out suite.jsonl
```

Generation is deterministic: the same seed reproduces the released suite byte-identically (verified on two machines with different Python versions).

## Reproducing the evaluation

```bash
python harness/evaluate_batched.py \
    --suite suite.jsonl \
    --hf-model "Qwen/Qwen2.5-7B-Instruct" \
    --greedy --seed 0 \
    --limit 300 --style cot --max-new-tokens 1400 \
    --batch-size 4 --tag cot300
# then build the tables:
python harness/evaluate_batched.py --suite suite.jsonl --merge cot300
```

Interrupted runs resume automatically from the raw prediction file.

## Headline results (step-by-step, 300-item subset, Qwen2.5-7B-Instruct 4-bit)

| | Arabic | English |
|---|---|---|
| Judgment accuracy | 0.607 [0.550–0.663] | 0.627 [0.573–0.680] |
| Step-localization | 0.396 [0.313–0.472] | 0.438 [0.354–0.521] |
| Constant-strategy ceiling | 0.164 | 0.164 |

Paired Arabic−English step-localization difference: **−0.042 [−0.132, +0.056]** — no cross-lingual gap detectable above the 0.135 bound of this subset. 0.0% unparsable responses. See the paper for the full protocol, analytic floors, and the detection-without-verdict failure mode (26% of responses).

## License

Code: MIT. Suite data (`suite.jsonl`) and results: CC BY 4.0.

## Citation

See `CITATION.cff`, or cite the Zenodo DOI badge above once minted.
