"""Streamlit app — Disaster Tweet Classifier.

Run locally:  streamlit run streamlit_app.py
"""
import io
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))  # lets the pickle import `features`
from features import clean_text, tokenize  # noqa: E402

st.set_page_config(page_title="Disaster Tweet Classifier", page_icon="🚨", layout="centered")

EXAMPLES = {
    "Wildfire evacuation": "13,000 people receive #wildfires evacuation orders in California",
    "Flood warning": "Flash flood warning issued for downtown, roads closed #flood",
    "Earthquake": "Earthquake of magnitude 6.1 strikes near coast, buildings collapsed",
    "Casual / metaphor": "This new album is the bomb lol",
}


@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    model = joblib.load(os.path.join(BASE, "models", "disaster_tweet_model.pkl"))
    meta = json.load(open(os.path.join(BASE, "models", "metadata.json")))
    keywords = json.load(open(os.path.join(BASE, "models", "keywords.json")))
    return model, meta, keywords


model, meta, keywords = load_assets()


def predict_df(texts, kws=None):
    kws = kws if kws is not None else [None] * len(texts)
    frame = pd.DataFrame({"text": list(texts), "keyword": [k or None for k in kws]})
    return model.predict_proba(frame)[:, 1]


st.title("🚨 Disaster Tweet Classifier")
st.caption("Does a tweet describe a **real disaster**? NLP model: TF-IDF + hand-crafted features + Linear SVM.")

tab_single, tab_batch, tab_model = st.tabs(["Classify a tweet", "Batch (CSV)", "About the model"])

# ---------------------------------------------------------------- single tweet
with tab_single:
    st.session_state.setdefault("tweet", "")
    cols = st.columns(len(EXAMPLES))
    for col, (label, txt) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["tweet"] = txt

    tweet = st.text_area("Tweet text", key="tweet", max_chars=1000, height=120,
                         placeholder="e.g. Forest fire near La Ronge Sask. Canada")
    keyword = st.selectbox("Keyword (optional)", [""] + keywords,
                           help="The dataset tags each tweet with a search keyword; leave blank if unknown.")
    threshold = st.slider("Decision threshold", 0.2, 0.8, 0.5, 0.05,
                          help="Lower = catch more disasters (higher recall, more false alarms). Higher = fewer false alarms.")

    if st.button("Classify", type="primary"):
        if not tweet.strip():
            st.warning("Please enter some tweet text.")
        else:
            p = float(predict_df([tweet], [keyword])[0])
            is_disaster = p >= threshold
            (st.error if is_disaster else st.success)(
                f"{'⚠️ Real disaster' if is_disaster else '✅ Not a disaster'}  —  P(disaster) = {p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            with st.expander("How the text was processed"):
                st.write("**Cleaned:**", clean_text(tweet))
                st.write("**Tokens used:**", tokenize(tweet))

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV or Excel file with a **text** column (optional **keyword** column).")
    up = st.file_uploader("File", type=["csv", "xlsx"])
    if up is not None:
        data = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        if "text" not in data.columns:
            st.error(f"Column 'text' not found. Columns present: {list(data.columns)}")
        else:
            if len(data) > 20000:
                st.warning("Only the first 20,000 rows are scored.")
                data = data.head(20000)
            kws = data["keyword"].tolist() if "keyword" in data.columns else None
            data["p_disaster"] = predict_df(data["text"].fillna("").astype(str), kws).round(4)
            data["prediction"] = (data["p_disaster"] >= 0.5).map({True: "disaster", False: "not disaster"})
            c1, c2 = st.columns(2)
            c1.metric("Tweets scored", f"{len(data):,}")
            c2.metric("Flagged as disaster", f"{(data.prediction == 'disaster').mean():.1%}")
            st.dataframe(data, use_container_width=True, height=320)
            st.download_button("Download predictions (CSV)", data.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- model card
with tab_model:
    m = meta["test_metrics"]
    st.subheader(meta["best_model"])
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.3f}")
    b.metric("Precision", f"{m['precision']:.3f}")
    c.metric("Recall", f"{m['recall']:.3f}")
    d.metric("F1", f"{m['f1']:.3f}")
    st.caption(f"Held-out test set: {meta['n_test']:,} tweets · ROC-AUC {m['roc_auc']:.3f}")
    st.markdown(
        "**Known limitations**\n"
        "- Bag-of-words cannot fully resolve metaphor/sarcasm (e.g. “I'm on fire today”).\n"
        "- Training labels are noisy (identical tweets carry conflicting labels), which caps accuracy.\n"
        "- Trained on 2015-era English tweets; slang and events since then may be handled poorly."
    )
    figs = [("reports/figures/06_confusion_matrix.png", "Confusion matrix"),
            ("reports/figures/07_roc_curves.png", "ROC curves"),
            ("reports/eda/14_log_odds.png", "Most discriminative words (EDA)")]
    for path, cap in figs:
        full = os.path.join(BASE, path)
        if os.path.exists(full):
            st.image(full, caption=cap)
