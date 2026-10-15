# Disaster Tweet Classification (NLP)

Classifies a tweet as a **real disaster** or **not**, with a Flask web interface.
Author: Ashish Kumar · Next Hikes

## Structure
```
data/twitter_disaster.xlsx      dataset (7,613 labelled tweets)
src/features.py                 cleaning, tokenisation, meta-features (shared by training + app)
src/train.py                    EDA -> features -> model selection -> evaluation -> pickle
notebooks/*.ipynb               same pipeline as a notebook (run from notebooks/)
models/                         disaster_tweet_model.pkl + metadata.json
reports/                        figures/, cv_results.csv, test_results.csv, top_errors.csv
app.py, templates/index.html    Flask app (UI + /predict JSON API)
```

## Run
```bash
pip install -r requirements.txt
python src/train.py        # retrains, regenerates figures + model
python app.py              # http://127.0.0.1:5000
```
API: `POST /predict {"text": "...", "keyword": "optional"}` -> `{label, verdict, probability, confidence}`

## Method
1. **Prep** - remove URLs/mentions/HTML/punctuation/digits, keep hashtag words, lowercase, tokenise, drop stop-words (keeping negations). Exact-duplicate tweets removed (179 rows shared duplicate text; 18 texts had conflicting labels and were dropped) to avoid train/test leakage. Stratified 80/20 split.
2. **Features** - TF-IDF (1-2 grams) on cleaned text + `keyword` token, plus length, hashtag, mention, URL, caps-ratio and punctuation features.
3. **Models** - Logistic Regression, Multinomial NB, Linear SVM (calibrated), Random Forest, MLP; 5-fold stratified CV, then grid search on LR and SVM.
4. **Deployment** - the whole pipeline (cleaning + features + classifier) is one pickle; Flask only calls `predict_proba`.

## Results (held-out test set, n=1,497)
| Model | Acc | Prec | Rec | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Linear SVM (tuned)** | 0.783 | 0.763 | 0.712 | 0.736 | 0.844 |
| Logistic Regression (tuned) | 0.778 | 0.744 | 0.729 | 0.736 | 0.843 |
| MLP | 0.790 | 0.837 | 0.629 | 0.718 | 0.842 |
| Multinomial NB | 0.789 | 0.843 | 0.621 | 0.715 | 0.838 |
| Random Forest | 0.758 | 0.797 | 0.580 | 0.672 | 0.816 |

## Limitations / next steps
- **Overfitting:** train F1 ~0.98 vs test F1 ~0.74. Stronger regularisation or more data would help.
- Bag-of-words cannot resolve metaphor ("I'm on fire today") - it can be pushed to "disaster" by keywords alone.
- Labels are noisy (see `reports/top_errors.csv`), which caps achievable accuracy.
- Next: pre-trained embeddings (GloVe/Word2Vec) or fine-tuning a transformer (DistilBERT/BERTweet) - typically the biggest gain on this task. Not included here because no deep-learning/embedding libraries were available in this environment.
- Deploy: `Procfile` is included for Heroku/Render (`gunicorn app:app`).
