# Datasets

This folder has the two datasets you'll use this summer. The summer plan calls them **ICSE26_DS** and **Mutation_DS**.

## ICSE26_DS — `icse26/`

The data from the ICSE 2026 paper. You only care about two benchmarks:

- `icse26/bigcodebench/`
- `icse26/humaneval/`

Ignore the other folders (`coref`, `ner`, `sentiment`, `translations`, …) — they are from unrelated experiments.

Each benchmark looks like this:

```
icse26/bigcodebench/
├── fold_0/ ... fold_9/      # the original "flat" format
│   ├── fit.jsonl            # train (used to learn the reading vector)
│   ├── validate.jsonl       # used to pick the layer
│   └── test.jsonl           # held-out evaluation
└── icse26/                  # SAME data, reformatted for the internal pipeline
    └── fold_*/...
```

The `fold_N/*.jsonl` files at the top are the simple format — one task per line with a list of candidate implementations and a parallel list of `0/1` labels (`1` = correct). The nested `icse26/` directory is a reformatted version that's more appropriate for this repository. THEY REPRESENT THE SAME DATA. To be clear, let's use an example:
- `dataset/icse26/bigcodebench/fold_0/fit.jsonl` has the same tasks and implementations as
- `dataset/icse26/bigcodebench/icse26/fold_0/fit.jsonl`

I'd say use the nested `icse26/`, it's a bit more verbose but more organized.

## Mutation_DS — `bigcodebench/`

The error-specific mutation + refactoring dataset. The two things to know:

- **`bigcodebench/frozen.jsonl`** (or its compressed `.zst`) — the full aggregated dataset. This is what "Mutation_DS" refers to. Each line is one task with its original/correct code, its mutants, and the LLM-refactored confounding variants.
- **`bigcodebench/phase2/sN_<name>/`** — pre-built splits over `frozen.jsonl` (different ways of partitioning the data into fit/validate/test). For Phase 2 (the analysis phase) start with one of these splits rather than the full file. Most relevant ones:
  - `s1_baseline/` — straightforward random split, good starting point.
  - `s2_test_confound/`, `s2b_val_test_confound/` — test sets focused on the confounding variants.
  - `s3_holdout_*/`, `s8_single_*/` — splits centered on one bug/operator type at a time.

### Working on Phase 1 (data generation)?

Look in `bigcodebench/intermediate/`. That's the output of the mutation + refactoring pipeline before it gets aggregated into `frozen.jsonl`:

- `tasks.jsonl` → original tasks
- `dropped_tasks.jsonl` → tasks discarded right away because the canonical/reference solution failed the test cases in our environment
- `mutants.jsonl`, `validated_mutants.jsonl` → mutants before/after running tests
- `confound_candidates.jsonl`, `validated_confounded.jsonl` → refactored variants before/after validation

If you're auditing or editing the pipeline, this is where the evidence lives.

---

If anything is unclear or seems out of date, please ask.
