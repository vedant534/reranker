# Intent-Aware E-commerce Search Reranker

This interview-sized project reranks product candidates from Amazon's ESCI Shopping Queries dataset. For each query, it learns to put Exact (`E`) products ahead of Substitutes (`S`), Complements (`C`), and Irrelevant (`I`) products.

This is **reranking over candidates already supplied by ESCI**, not a retrieval engine. It cannot return a product that is missing from the candidate set.

## Approach

The pipeline uses the official reduced Task 1 data (`small_version == 1`), US locale, and official train/test split. Validation queries are sampled only from the official training split, and all sampling is query-disjoint.

It compares:

1. word TF-IDF cosine similarity between query and title;
2. a validation-selected weighted average of word and character TF-IDF similarities;
3. a LightGBM LambdaRank challenger using 13 straightforward lexical and metadata-match features.

The TF-IDF vocabulary is fitted only on unique sampled training queries plus unique sampled training titles. ESCI labels, split, source, IDs, and row order are never model features.

The main metric is nDCG@10 using gains `I=0`, `C=0.01`, `S=0.1`, and `E=1.0`. This heavily rewards Exact results while retaining a small distinction between Substitute, Complement, and Irrelevant results. The reports also include Exact MRR@10, Exact-or-Substitute Recall@5, Complement-or-Irrelevant Exposure@5, and warmed-up end-to-end p50/p95 scoring latency.

## Setup and data

Python 3.9–3.12 is supported by the pinned dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place these official files in `data/`:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`

They can be downloaded from the [Amazon ESCI repository](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset). Existing files do not need to be downloaded again.

```bash
mkdir -p data
test -f data/shopping_queries_dataset_examples.parquet || curl -L --fail -o data/shopping_queries_dataset_examples.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
test -f data/shopping_queries_dataset_products.parquet || curl -L --fail -o data/shopping_queries_dataset_products.parquet https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet
```

## Run

```bash
python run_pipeline.py --config config.yaml
python -m pytest -q
streamlit run app.py --server.headless true -- --config config.yaml
```

The pipeline saves the ranker, TF-IDF preprocessing, frozen low-overlap threshold, and test predictions under `artifacts/`. Metrics, CSV comparisons, and plots are written under `reports/`.

## Verified sampled results

The real-data run used 5,000 training, 1,000 validation, and 1,000 official-test queries (100,787 / 20,207 / 20,013 candidate rows). The validation-selected lexical word weight was 0.25, and the LightGBM model stopped at iteration 89.

| Model | nDCG@10 | Exact MRR@10 | E/S Recall@5 | C/I Exposure@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Word TF-IDF | 0.6668 | 0.7139 | 0.3087 | 0.1790 | 3.20 | 5.93 |
| Combined lexical | 0.6889 | 0.7409 | 0.3117 | 0.1732 | 3.22 | 5.96 |
| LightGBM ranker | **0.7219** | **0.7914** | **0.3159** | **0.1620** | 3.48 | 6.24 |

The challenger improved nDCG@10 by 0.0330 and Exact MRR@10 by 0.0505 over the combined lexical baseline while reducing C/I Exposure@5 by 0.0112. These are offline relevance results for the sampled candidate-ranking task, not retrieval or business-impact measurements. Full precision results are in [`reports/model_comparison.csv`](reports/model_comparison.csv) and [`reports/metrics.json`](reports/metrics.json).

![Model comparison](reports/model_comparison.png)

![Feature importance](reports/feature_importance.png)

## Error-analysis slices

`reports/query_slice_metrics.csv` covers one/two-token queries, queries containing numeric or model tokens, queries of five or more tokens, and low-overlap queries. Low overlap is defined by mean candidate query-token coverage at or below the validation-set bottom quartile; that validation threshold is frozen before test evaluation.

- Numeric/model-token queries showed the largest challenger gain: nDCG@10 rose from 0.6257 to 0.7058 across 210 queries.
- Long queries improved from 0.6609 to 0.6894 across 315 queries.
- Low-overlap queries improved only from 0.6716 to 0.6802 across 240 queries, indicating that limited lexical evidence remains the clearest weak slice.

## Limitations

- Candidate products are supplied by the dataset, so retrieval recall is not measured.
- ESCI labels are relevance judgments, not clicks, conversions, or revenue.
- The model does not use live price, availability, inventory, or personalization.
- Only the US English subset is evaluated.
- Offline relevance improvements do not guarantee online customer or business impact.
