from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "news_articles.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].fillna("").astype(str).str.strip()
        + " "
        + df["article_text"].fillna("").astype(str).str.strip()
    )


def train_models() -> tuple[TfidfVectorizer, KMeans, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH, parse_dates=["published_at"])
    text = build_text(df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1500,
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(text)

    model = KMeans(n_clusters=8, random_state=42, n_init=20)
    df["cluster"] = model.fit_predict(matrix)
    return vectorizer, model, df


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    vectorizer, model, df = train_models()

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(model, MODELS_DIR / "kmeans_model.joblib", compress=3)

    terms = vectorizer.get_feature_names_out()
    keyword_rows = []
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[-10:][::-1]
        keyword_rows.append(
            {
                "cluster": cluster_id,
                "keywords": ", ".join(terms[top_indices]),
            }
        )

    pd.DataFrame(keyword_rows).to_csv(
        REPORTS_DIR / "topic_keywords.csv", index=False
    )
    df.to_csv(REPORTS_DIR / "articles_with_clusters.csv", index=False)

    metadata = {
        "articles_used": int(len(df)),
        "clusters": int(model.n_clusters),
        "tfidf_features": int(len(terms)),
        "random_state": 42,
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
