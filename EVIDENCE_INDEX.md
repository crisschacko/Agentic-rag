# Agentic RAG Project — Interview Evidence Index

**Authors**
- Ms. Adithya Vinod
- Mr. Criss Chacko Chirayath Staby Chacko

## Verification status

- [x] Author names standardized across project text artifacts.
- [x] Historical experiment CSV preserved unchanged.
- [x] Historical aggregate values preserved (10/10 retrieval attempts = 2; 10/10 final verification = INSUFFICIENT).
- [x] The 4/10 vs 10/10 trace discrepancy is explicitly preserved and itemized rather than silently rewritten.
- [x] Current packaging/reproduction environment archived.
- [ ] Exact historical corpus checksum archived — **blocked because the exact 19-page corpus file is absent from the supplied package**.
- [ ] Complete per-question retrieval rollout trace reconciled with aggregate CSV — **blocked because the supplied package has no authoritative per-question rollout log**.
- [ ] Manual correctness annotations complete — **blocked because the supplied manual evaluation sheet is blank**.
- [ ] Manual evidence-support annotations complete — **blocked because the retrieved evidence passages are not archived per question**.

## Research-integrity rule

No missing evidence has been fabricated. In particular, `INSUFFICIENT` has not been changed to `SUFFICIENT`, and blank manual annotations have not been converted into positive scores.

## What closes the remaining items

1. Add the exact historical 19-page corpus and record SHA-256.
2. Re-run the experiment from a clean environment and save a per-question rollout trace containing every retrieval round, verification decision, reformulated query, retrieved chunk IDs/text, final decision, and timing.
3. Reconcile that trace against the aggregate CSV and regenerate summary CSVs from the raw trace.
4. Have the two authors independently annotate correctness and evidence support for all 10 questions using a fixed rubric, then archive the signed/dated annotation sheet and adjudication record.

This package is therefore **interview-ready as an auditable research artifact**, but it does not falsely claim that unavailable historical evidence or manual annotations exist.
