from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "articles_with_clusters.csv"
REPORTS = ROOT / "reports"


def detect_spikes(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for category, group in daily.groupby("category", sort=False):
        g = group.sort_values("date").copy()
        g["rolling_mean"] = g["article_count"].shift(1).rolling(14, min_periods=7).mean()
        g["rolling_std"] = g["article_count"].shift(1).rolling(14, min_periods=7).std().fillna(0)
        g["threshold"] = g["rolling_mean"] + 2.0 * g["rolling_std"]
        g["anomaly_score"] = (
            (g["article_count"] - g["rolling_mean"])
            / g["rolling_std"].replace(0, 1)
        )
        frames.append(g)

    scored = pd.concat(frames, ignore_index=True).dropna(subset=["rolling_mean"])
    spikes = scored.sort_values(
        ["anomaly_score", "article_count"], ascending=False
    ).head(42).copy()
    spikes["is_spike"] = True
    return spikes.sort_values("date")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["published_at"])

    daily = (
        df.assign(date=df["published_at"].dt.date)
        .groupby(["date", "category"], as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
    )
    full_dates = pd.MultiIndex.from_product(
        [
            pd.date_range(
                df["published_at"].min().date(),
                df["published_at"].max().date(),
                freq="D",
            ).date,
            sorted(df["category"].unique()),
        ],
        names=["date", "category"],
    ).to_frame(index=False)
    daily = full_dates.merge(daily, how="left", on=["date", "category"]).fillna(
        {"article_count": 0}
    )
    daily["article_count"] = daily["article_count"].astype(int)
    daily.to_csv(REPORTS / "daily_topic_volume.csv", index=False)

    spikes = detect_spikes(daily)
    spikes.to_csv(REPORTS / "detected_spikes.csv", index=False)

    brand = (
        df[df["brand_mentioned"] != "None"]
        .assign(date=lambda x: x["published_at"].dt.date)
        .groupby(["date", "brand_mentioned"], as_index=False)
        .size()
        .rename(columns={"size": "mention_count", "brand_mentioned": "brand"})
    )
    brand.to_csv(REPORTS / "brand_trends.csv", index=False)

    summary = {
        "article_count": int(len(df)),
        "category_count": int(df["category"].nunique()),
        "cluster_count": int(df["cluster"].nunique()),
        "detected_spikes": int(len(spikes)),
        "tracked_brands": int(brand["brand"].nunique()),
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (REPORTS / "summary_report.md").write_text(
        "# News Trend Detection Summary\n\n"
        f"- Articles analyzed: **{summary['article_count']:,}**\n"
        f"- Broad categories: **{summary['category_count']}**\n"
        f"- TF-IDF/K-Means clusters: **{summary['cluster_count']}**\n"
        f"- Unusual topic-volume spikes: **{summary['detected_spikes']}**\n"
        f"- Brands tracked: **{summary['tracked_brands']}**\n\n"
        "The project combines NLP clustering, keyword extraction, rolling time-series baselines, "
        "and brand-mention monitoring to support early trend discovery.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
