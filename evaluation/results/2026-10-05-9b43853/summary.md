# Evaluation: whitepaper_eval.json

- Run: 2026-10-06T12:50:41 to 2026-10-06T12:53:39 against http://localhost:8000 (app commit 9b43853)
- App model: gemini-flash-lite-latest; judge: gemini-3.5-flash (temperature 0)
- Judge requests sent: 11 of a cap of 20
- Context metrics use each answer's final retrieval, recreated with the app's code (k=4; complex: k per sub-query, merged)

## Answers

### In-scope, factual and multi-passage (n=19, scored 19)

| Metric | Value |
|---|---|
| Correct / partial / incorrect | 17 / 2 / 0 |
| Accuracy (correct only) | 0.89 |
| Score (correct = 1, partial = 0.5) | 0.95 |
| Faithfulness (mean share of supported claims) | 1.00 |
| Answer relevance (relevant / partial / irrelevant) | 19 / 0 / 0 |
| Context recall (mean share of supporting quotes retrieved) | 0.98 |
| Context recall (all supporting quotes retrieved) | 18/19 |
| Context precision (mean share of retrieved chunks judged relevant) | 0.37 |
| Errors | 0 |
| Unjudged | 0 |

### Synthesis (q20, reported separately) (n=1, scored 1)

| Metric | Value |
|---|---|
| Correct / partial / incorrect | 0 / 1 / 0 |
| Accuracy (correct only) | 0.00 |
| Score (correct = 1, partial = 0.5) | 0.50 |
| Faithfulness (mean share of supported claims) | 1.00 |
| Answer relevance (relevant / partial / irrelevant) | 1 / 0 / 0 |
| Context recall (mean share of supporting quotes retrieved) | 0.00 |
| Context recall (all supporting quotes retrieved) | 0/1 |
| Context precision (mean share of retrieved chunks judged relevant) | 0.57 |
| Errors | 0 |
| Unjudged | 0 |

## By path (router label)

In-scope:

| Path | n | Correct / partial / incorrect | Faithfulness | Recall | Precision | p50 latency, s | Avg rewrites | Regenerated |
|---|---|---|---|---|---|---|---|---|
| simple | 13 | 12/1/0 | 1.00 | 1.00 | 0.37 | 18.2 | 0.00 | 0 |
| complex | 7 | 5/2/0 | 1.00 | 0.81 | 0.41 | 23.3 | 0.00 | 0 |

Out-of-scope router labels: oos01=simple, oos02=simple, oos03=simple, oos04=simple, oos05=simple

## Refusals

Out-of-scope refused with no_relevant_docs: 5/5

## Self-correction

- Answers regenerated: 0/25
- Grounded after regeneration: 0
- Still ungrounded, returned with a caveat: 0

## Hallucination check agreement

App's final `is_grounded` vs judge (grounded = every claim supported):

| | judge grounded | judge not grounded |
|---|---|---|
| app is_grounded = true | 20 | 0 |
| app is_grounded = false | 0 | 0 |

Agreement: 20/20 (100%)

## Latency under evaluation load and rewrites

Queries are sent back to back, so the app's rate limiter queues calls (about 5 s each once the burst of 3 is used); see latency_probe.md for single-user latency.

| | p50 | p95 |
|---|---|---|
| Server latency, s (n=25) | 22.6 | 38.7 |
| Client wall time, s (n=25) | 22.6 | 38.7 |

- Average rewrites, in-scope: 0.00; out-of-scope: 3.00
- Errors (status error or HTTP failure): 0

## Not fully correct, failures, caveats, refusal misses and unjudged

### q12 (factual, path simple): Which similarity metric does the whitepaper say works well for lower dimensions, and what about higher-dimensional data?

- Outcome: partial, status success, 0 rewrites, revised=False, is_grounded=True
- Final query: 'Which similarity metric does the whitepaper say works well for lower dimensions, and what about higher-dimensional data?'; sub-queries: []
- Mechanical diagnosis: generation: supporting passages were in context
- Judge: The system answer correctly identifies Euclidean distance for lower dimensions and dot product for higher-dimensional data. However, it misses cosine similarity, which the reference answer (and the source text) also associates with working better for higher-dimensional data (as cosine similarity and dot product are equivalent when vector norms are 1).
- Answer: Based on the provided text, Euclidean distance works well for lower dimensions, while the dot product seems to work better for higher-dimensional data.

### q13 (multi_passage, path complex): How does the runtime of linear search compare with approximate nearest neighbour (ANN) search?

- Outcome: partial, status success, 0 rewrites, revised=False, is_grounded=True
- Final query: 'How does the runtime of linear search compare with approximate nearest neighbour (ANN) search?'; sub-queries: ['runtime complexity of linear search', 'runtime complexity of approximate nearest neighbor ANN search']
- Mechanical diagnosis: generation: supporting passages were in context
- Judge: The system answer correctly states that linear search scales linearly and is too slow for large datasets, and that ANN reduces the search space and computational requirements. However, it misses the specific complexity of ANN (O(log N)) and the fact that ANN comes with a small margin of error, which are key parts of the reference answer.
- Answer: Traditional linear search scales linearly with the number of documents, making it too slow for large datasets, whereas approximate nearest neighbour search greatly reduces the search space and computational requirements for much faster performance.

### q20 (synthesis, path complex): How do sparse and dense retrieval differ?

- Outcome: partial, status success, 0 rewrites, revised=False, is_grounded=True
- Final query: 'How do sparse and dense retrieval differ?'; sub-queries: ['How does sparse retrieval work', 'How does dense retrieval work', 'Differences between sparse and dense retrieval']
- Mechanical diagnosis: retrieval miss: no supporting passage in the final retrieval
- Judge: The system answer correctly identifies the core differences between sparse (term-based, ignores word order and semantics) and dense retrieval (uses embeddings for semantic similarity, handles different wording). However, it is marked as partial because it misses some key facts from the reference answer, specifically that dense retrieval can be weak on literal terms like IDs, and that combining sparse and dense search addresses these limitations.
- Answer: Sparse retrieval relies on traditional term-based matching approaches that ignore word order and deeper semantic meanings. In contrast, dense retrieval uses embeddings to capture semantic similarity, allowing systems to find relevant results and make accurate recommendations even when the wording of the query differs completely from the documents.

