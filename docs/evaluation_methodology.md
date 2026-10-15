# Evaluation methodology

Every metric below is computed from real execution on the synthetic dataset. Nothing is fabricated.

## Structural validity

- **Valid JSON rate**: share of outputs containing a parseable JSON object.
- **Schema compliance rate**: share whose structured object validates against `CareSummary` (Pydantic).
- **Invalid/missing evidence references**: counted during unsupported-claim analysis.

## Fact preservation (heuristic)

For each example, reference `expected_facts` are checked against the output text: a fact is "recalled" when all of its content words (length > 2, after Arabic normalization) appear in the output. This is a **heuristic** — semantic equivalence is not proven by word overlap.

- **Important-fact recall**: mean per-example recall.
- **Omitted important facts**: examples where recall < 1.0.
- Precision is measured only where labels allow reliable matching.

Arabic normalization handles alef variants, teh marbuta, tatweel, diacritics, and punctuation before matching — exact string matching alone is not used.

## Reliability flags

- **Unsupported-claim rate (heuristic)**: claims whose evidence quote is missing or not traceable to the input note.
- **Uncertainty preservation**: share of reference-uncertain examples where the output marks observations uncertain or lists unclear/conflicting items.
- **Contradiction preservation**: same, for reference-contradictory examples.

## Performance

- Latency per example and totals; failures and timeouts; valid/invalid JSON split.

## Human review

Labels: Correct / Partially correct / Incorrect / Contains unsupported claims / Loses uncertainty / Omits important information. Automated metrics cannot replace human judgment for semantic correctness — review Tab 4 and export results.

## What is NOT measured

- Clinical correctness (this tool is not medical).
- Real-world caregiver behavior (data is synthetic).
- Anything on examples used to design prompts (the dataset is fixed; hold out the test split mentally).
