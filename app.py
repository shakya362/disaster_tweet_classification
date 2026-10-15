"""Flask web app for disaster-tweet classification.

Run:  python app.py   ->  http://127.0.0.1:5000
API:  POST /predict  {"text": "...", "keyword": "optional"}
"""
import os, sys, json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))   # so the pickle can import `features`

app = Flask(__name__)
model = joblib.load(os.path.join(BASE, "models", "disaster_tweet_model.pkl"))
meta = json.load(open(os.path.join(BASE, "models", "metadata.json")))
THRESHOLD = 0.5


def classify(text: str, keyword: str = ""):
    row = pd.DataFrame({"text": [text], "keyword": [keyword or None]})
    p = float(model.predict_proba(row)[0, 1])
    return {"label": int(p >= THRESHOLD),
            "verdict": "Real disaster" if p >= THRESHOLD else "Not a disaster",
            "probability": round(p, 4),
            "confidence": round(max(p, 1 - p), 4)}


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Please enter some tweet text."), 400
    if len(text) > 1000:
        return jsonify(error="Text too long (max 1000 characters)."), 400
    return jsonify(classify(text, (data.get("keyword") or "").strip()))


@app.route("/health")
def health():
    return jsonify(status="ok", model=meta["best_model"])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
