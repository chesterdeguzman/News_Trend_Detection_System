# News Trend Detection System

An NLP and time-series portfolio project for detecting emerging news topics, tracking brand mentions, and identifying unusual increases in article volume.

## Project highlights

- Prepared a **5,121-article educational dataset**
- Vectorized article text using **TF-IDF**
- Clustered related stories into **8 topic clusters** with K-Means
- Extracted representative keywords for every cluster
- Built rolling baselines for daily category volume
- Flagged **42 unusual topic-volume spikes**
- Tracked mentions for eight well-known brands
- Built an interactive **Streamlit dashboard**

## Business value

The project demonstrates how NLP, unsupervised learning, and anomaly detection can support media intelligence, brand monitoring, and early trend discovery.

## Repository structure

```text
news-trend-detection-system/
├── app.py
├── data/
│   ├── news_articles.csv
│   └── sample_articles.csv
├── models/
│   ├── kmeans_model.joblib
│   ├── model_metadata.json
│   └── tfidf_vectorizer.joblib
├── notebooks/
│   └── trend_detection.ipynb
├── reports/
│   ├── articles_with_clusters.csv
│   ├── brand_trends.csv
│   ├── daily_topic_volume.csv
│   ├── detected_spikes.csv
│   ├── metrics.json
│   ├── summary_report.md
│   └── topic_keywords.csv
├── src/
│   ├── analyze.py
│   ├── generate_data.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── LICENSE
└── requirements.txt
```

## Installation

```bash
git clone <your-repository-url>
cd news-trend-detection-system
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the project

Generate the educational dataset:

```bash
python src/generate_data.py
```

Train and save the TF-IDF and K-Means artifacts:

```bash
python src/train.py
```

Run the time-series and brand trend analysis:

```bash
python src/analyze.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Methods

### Topic discovery
Article titles and body text are combined and transformed with TF-IDF using unigram and bigram features. K-Means groups semantically related stories into eight broad clusters. The highest-weight TF-IDF terms summarize each cluster.

### Trend detection
Articles are aggregated by day and category. A 14-day rolling mean and standard deviation provide a historical baseline. The strongest positive deviations are retained as the educational anomaly set.

### Brand monitoring
Explicit brand references are aggregated by date, allowing dashboard users to compare mention volume and identify changes in attention.

## Notes

- The dataset is synthetic and intended for education and portfolio demonstration.
- The trained TF-IDF and K-Means artifacts are included in `models/` and total less than 25 KB.
- Run `python src/train.py` whenever you want to regenerate the saved models.
- The repository remains lightweight and GitHub-friendly.

## License

MIT License.
