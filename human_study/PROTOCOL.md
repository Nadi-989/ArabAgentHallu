# Bilingual human naturalness study — protocol (prepared, pending execution)

**Goal.** Human assessment of (a) surface naturalness of the Arabic and English
renderings and (b) a redundant human check of semantic equivalence. Note that
semantic equivalence already holds *by construction* — both surfaces are
rendered from the same executable program — and the structural AR/EN mirror is
verified mechanically for all 900 items by `verify_suite.py`; the human arm
targets naturalness, which no mechanical check can attest.

**Sample.** 60 pairs from `suite.jsonl`: 30 hallucinated (6 per category) +
30 clean, drawn with seed 13 (reproducible), presented in shuffled order.
The exact sample is embedded in `ArabAgentHallu_rater_form.xlsx`.

**Raters.** Two Arabic–English bilinguals, working fully independently.

**Procedure.** Each rater receives a private copy of the form, reads the
instruction sheet, and fills four columns per pair: Arabic naturalness (1–5),
English naturalness (1–5), semantic equivalence (yes/no), optional note.
Estimated effort: 60–90 minutes per rater.

**Analysis.** `python compute_agreement.py rater1.xlsx rater2.xlsx`
reports per-rater means ± sd, equivalence rate, Cohen's kappa on equivalence,
inter-rater Pearson correlation on naturalness, and a summary sentence.

**Status.** Instrument and analysis released here; execution is the first item
of follow-up work (see paper, Section 4.11).
