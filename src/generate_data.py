from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "news_articles.csv"
SAMPLE_PATH = ROOT / "data" / "sample_articles.csv"

CATEGORIES = {
    "Technology": ["artificial intelligence", "cloud", "chip", "software", "cybersecurity", "startup"],
    "Business": ["earnings", "merger", "market", "investment", "retail", "strategy"],
    "Health": ["hospital", "vaccine", "wellness", "clinical", "medicine", "public health"],
    "Politics": ["election", "policy", "senate", "government", "campaign", "regulation"],
    "Sports": ["championship", "league", "athlete", "coach", "tournament", "season"],
    "Entertainment": ["film", "music", "streaming", "celebrity", "festival", "television"],
    "Science": ["research", "space", "climate", "discovery", "laboratory", "energy"],
    "World": ["diplomacy", "trade", "conflict", "summit", "region", "humanitarian"],
}
BRANDS = ["OpenAI", "Apple", "Google", "Microsoft", "Tesla", "Nike", "Netflix", "Amazon"]


def build_dataset(n_rows: int = 5121, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    category_names = list(CATEGORIES)

    # Create 42 category-day spikes by boosting sampling weights.
    spike_pairs: set[tuple[pd.Timestamp, str]] = set()
    while len(spike_pairs) < 42:
        spike_pairs.add((pd.Timestamp(random.choice(dates)), random.choice(category_names)))

    weights = []
    pairs = []
    for date in dates:
        for category in category_names:
            pairs.append((date, category))
            base = 1.0
            if (pd.Timestamp(date), category) in spike_pairs:
                base = 7.5
            weights.append(base)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    selected = rng.choice(len(pairs), size=n_rows, p=weights)

    rows = []
    outlets = ["Daily Ledger", "Global Wire", "Metro Journal", "Insight News", "Morning Brief"]
    for idx, pair_idx in enumerate(selected, start=1):
        date, category = pairs[pair_idx]
        terms = random.sample(CATEGORIES[category], k=3)
        brand = random.choice(BRANDS) if rng.random() < 0.38 else None
        brand_phrase = f" involving {brand}" if brand else ""
        title = f"{terms[0].title()} developments reshape {category.lower()} outlook{brand_phrase}"
        text = (
            f"A new {terms[0]} report highlights {terms[1]} and {terms[2]} developments in the "
            f"{category.lower()} sector{brand_phrase}. Analysts discuss possible effects on audiences, "
            "organizations, and future planning."
        )
        rows.append({
            "article_id": f"NEWS-{idx:05d}",
            "published_at": pd.Timestamp(date) + pd.Timedelta(hours=int(rng.integers(0, 24))),
            "source": random.choice(outlets),
            "category": category,
            "title": title,
            "article_text": text,
            "brand_mentioned": brand or "None",
        })

    df = pd.DataFrame(rows).sort_values("published_at").reset_index(drop=True)
    return df


def main() -> None:
    df = build_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    df.head(100).to_csv(SAMPLE_PATH, index=False)
    print(f"Saved {len(df):,} articles to {DATA_PATH}")


if __name__ == "__main__":
    main()
