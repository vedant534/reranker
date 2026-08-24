import argparse
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data import load_config
from src.predict import predict_reranked


def parse_config_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.yaml")
    arguments, _ = parser.parse_known_args()
    return arguments.config


@st.cache_resource
def load_artifacts(config_path):
    config = load_config(config_path)
    artifact_dir = Path(config["paths"]["artifact_dir"])
    bundle = joblib.load(artifact_dir / "feature_bundle.joblib")
    ranker = joblib.load(artifact_dir / "ranker.joblib")
    return config, bundle, ranker


@st.cache_data
def load_test_predictions(artifact_dir):
    return pd.read_parquet(Path(artifact_dir) / "final_test_predictions.parquet")


st.set_page_config(page_title="Graded-relevance reranking demo", layout="wide")
st.title("Graded-relevance e-commerce search reranker")
st.caption(
    "This demo reranks candidate products already supplied by ESCI. "
    "It is not a product retrieval or general search engine."
)

config_path = parse_config_path()
preflight_config = load_config(config_path)
preflight_artifact_dir = Path(preflight_config["paths"]["artifact_dir"])
required_artifacts = [
    preflight_artifact_dir / "feature_bundle.joblib",
    preflight_artifact_dir / "ranker.joblib",
    preflight_artifact_dir / "final_test_predictions.parquet",
]
missing_artifacts = [path.name for path in required_artifacts if not path.exists()]
if missing_artifacts:
    st.error(
        "The demo artifacts are not ready yet. Missing: "
        + ", ".join(missing_artifacts)
        + "."
    )
    st.write("Run the pipeline once, then restart this app:")
    st.code(f"python run_pipeline.py --config {config_path}")
    st.stop()

config, feature_bundle, ranker = load_artifacts(config_path)
test_data = load_test_predictions(config["paths"]["artifact_dir"])
queries = test_data[["query_id", "query"]].drop_duplicates().sort_values("query_id")
query_options = {
    f"{row.query_id}: {row.query}": row.query_id for row in queries.itertuples(index=False)
}
selection = st.selectbox("Choose a saved test query", query_options)
query_id = query_options[selection]
candidates = test_data[test_data["query_id"] == query_id].copy()
query = candidates["query"].iloc[0]
reranked = predict_reranked(query, candidates, feature_bundle, ranker)

columns = [
    "product_title",
    "product_brand",
    "lexical_score",
    "predicted_score",
    "lexical_rank",
    "model_rank",
    "esci_label",
]
left, right = st.columns(2)
with left:
    st.subheader("Combined lexical baseline")
    st.dataframe(
        reranked.sort_values("lexical_rank")[columns],
        hide_index=True,
        use_container_width=True,
    )
with right:
    st.subheader("LightGBM reranker")
    st.dataframe(
        reranked.sort_values("model_rank")[columns],
        hide_index=True,
        use_container_width=True,
    )
