# Design notes

This directory holds longer-form design notes that don't fit in `ARCHITECTURE.md`,
`EVALUATION.md`, or `BENCHMARKS.md`. The current documents are intentionally
short — each design decision that needed justification was placed next to the
code that implements it (as docstrings) or in the corresponding top-level MD file.

| Topic | Where to look |
|-------|---------------|
| Hybrid fusion math | `ARCHITECTURE.md` §5 + `app/retrieval/hybrid.py` |
| Citation validation rules | `ARCHITECTURE.md` §8 + `app/generation/prompt.py` |
| Cost-estimate contract | `ARCHITECTURE.md` §9 + `app/generation/service.py:estimate_cost` |
| Cache key construction | `ARCHITECTURE.md` §10 + `app/caching/registry.py:stable_key` |
| Error taxonomy | `ARCHITECTURE.md` §13 + `app/domain/errors.py` |
| Benchmark methodology | `BENCHMARKS.md` §4 + `benchmarks/runner.py` |
| Evaluation metrics | `EVALUATION.md` §1–2 + `app/evaluation/metrics.py` |
