from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
KEYWORDS_PATH = ROOT / "reports" / "topic_keywords.csv"


def load_artifacts():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "kmeans_model.joblib")
    keywords = pd.read_csv(KEYWORDS_PATH).set_index("cluster")["keywords"].to_dict()
    return vectorizer, model, keywords


def predict_cluster(title: str, article_text: str) -> dict[str, object]:
    vectorizer, model, keywords = load_artifacts()
    combined = f"{title.strip()} {article_text.strip()}".strip()
    if not combined:
        raise ValueError("Provide a title, article text, or both.")

    matrix = vectorizer.transform([combined])
    cluster = int(model.predict(matrix)[0])
    distance = float(model.transform(matrix)[0][cluster])
    return {
        "cluster": cluster,
        "keywords": keywords.get(cluster, ""),
        "distance_to_cluster_center": round(distance, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign a news article to a topic cluster.")
    parser.add_argument("--title", default="", help="Article title")
    parser.add_argument("--text", default="", help="Article body text")
    args = parser.parse_args()
    print(predict_cluster(args.title, args.text))


if __name__ == "__main__":
    main()
