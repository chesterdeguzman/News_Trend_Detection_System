from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from predict import predict_cluster

st.set_page_config(page_title="News Trend Detection System", layout="wide")
st.title("News Trend Detection System")
st.caption("Explore emerging topics, article-volume anomalies, brand mentions, and new-article cluster predictions.")

@st.cache_data
def load_data():
    articles = pd.read_csv(REPORTS / "articles_with_clusters.csv", parse_dates=["published_at"])
    daily = pd.read_csv(REPORTS / "daily_topic_volume.csv", parse_dates=["date"])
    spikes = pd.read_csv(REPORTS / "detected_spikes.csv", parse_dates=["date"])
    brands = pd.read_csv(REPORTS / "brand_trends.csv", parse_dates=["date"])
    keywords = pd.read_csv(REPORTS / "topic_keywords.csv")
    return articles, daily, spikes, brands, keywords

articles, daily, spikes, brands, keywords = load_data()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles", f"{len(articles):,}")
c2.metric("Categories", articles["category"].nunique())
c3.metric("Detected spikes", len(spikes))
c4.metric("Tracked brands", brands["brand"].nunique())

page = st.sidebar.radio(
    "View",
    ["Topic Trends", "Detected Spikes", "Brand Mentions", "Article Explorer", "Cluster Keywords", "Predict New Article"],
)

if page == "Topic Trends":
    selected = st.multiselect("Categories", sorted(daily["category"].unique()), default=sorted(daily["category"].unique())[:4])
    view = daily[daily["category"].isin(selected)]
    st.plotly_chart(px.line(view, x="date", y="article_count", color="category", title="Daily topic volume"), use_container_width=True)
elif page == "Detected Spikes":
    st.plotly_chart(px.scatter(spikes, x="date", y="article_count", color="category", size="anomaly_score", hover_data=["rolling_mean", "threshold"], title="Unusual topic-volume spikes"), use_container_width=True)
    st.dataframe(spikes, use_container_width=True, hide_index=True)
elif page == "Brand Mentions":
    selected = st.multiselect("Brands", sorted(brands["brand"].unique()), default=sorted(brands["brand"].unique())[:4])
    view = brands[brands["brand"].isin(selected)]
    st.plotly_chart(px.line(view, x="date", y="mention_count", color="brand", title="Brand mentions over time"), use_container_width=True)
elif page == "Article Explorer":
    query = st.text_input("Search titles and article text")
    category = st.selectbox("Category", ["All"] + sorted(articles["category"].unique()))
    view = articles.copy()
    if query:
        mask = view["title"].str.contains(query, case=False, na=False) | view["article_text"].str.contains(query, case=False, na=False)
        view = view[mask]
    if category != "All":
        view = view[view["category"] == category]
    st.dataframe(view[["published_at", "source", "category", "cluster", "title", "brand_mentioned"]], use_container_width=True, hide_index=True)
elif page == "Cluster Keywords":
    st.dataframe(keywords, use_container_width=True, hide_index=True)
else:
    st.subheader("Assign a new article to a topic cluster")
    title = st.text_input("Article title")
    article_text = st.text_area("Article text", height=180)
    if st.button("Predict cluster", type="primary"):
        try:
            result = predict_cluster(title, article_text)
            st.success(f"Predicted cluster: {result['cluster']}")
            st.write(f"**Representative keywords:** {result['keywords']}")
            st.write(f"**Distance to cluster center:** {result['distance_to_cluster_center']}")
        except ValueError as exc:
            st.warning(str(exc))
