import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, brier_score_loss)
from .data import FEATURES


def chronological_split(data):
    dates = sorted(data.prediction_date.unique())
    if len(dates) < 10:
        raise ValueError("Need at least 10 distinct prediction dates for chronological evaluation.")
    val_start, test_start = dates[int(len(dates)*.6)], dates[int(len(dates)*.8)]
    train = data[(data.prediction_date < val_start) & (data.target_end < val_start)]
    val = data[(data.prediction_date >= val_start) & (data.prediction_date < test_start) &
               (data.target_end < test_start)]
    test = data[data.prediction_date >= test_start]
    for name, frame in [("training", train), ("validation", val), ("test", test)]:
        if len(frame) < 5 or frame.target.nunique() != 2:
            raise ValueError(f"{name} needs at least five events and both outcome classes; expand the dataset.")
    return train, val, test


def metrics(y, probabilities, threshold):
    predictions = np.asarray(probabilities) >= threshold
    both = len(np.unique(y)) == 2
    return dict(roc_auc=float(roc_auc_score(y, probabilities)) if both else None,
                average_precision=float(average_precision_score(y, probabilities)) if both else None,
                precision=float(precision_score(y, predictions, zero_division=0)),
                recall=float(recall_score(y, predictions, zero_division=0)),
                f1=float(f1_score(y, predictions, zero_division=0)),
                brier=float(brier_score_loss(y, probabilities)),
                confusion_matrix=confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
                n=len(y), prevalence=float(np.mean(y)))


def fit_models(train, val):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier
    models = {
        "baseline": make_pipeline(SimpleImputer(keep_empty_features=True), DummyClassifier(strategy="prior")),
        "logistic": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
        "xgboost": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  XGBClassifier(n_estimators=120, max_depth=2, learning_rate=.03,
                                                min_child_weight=5, subsample=.8, colsample_bytree=.8,
                                                reg_lambda=5, eval_metric="logloss", n_jobs=2, random_state=42))}
    scores, thresholds = {}, {}
    for name, model in models.items():
        model.fit(train[FEATURES], train.target.astype(int))
        p = model.predict_proba(val[FEATURES])[:, 1]
        candidates = np.linspace(.1, .9, 81)
        threshold = float(max(candidates, key=lambda t: (f1_score(val.target, p >= t), -abs(t-.5))))
        thresholds[name] = threshold
        scores[name] = metrics(val.target, p, threshold)
    # Baseline participates in selection: do not claim ML improvement if none exists.
    winner = max(scores, key=lambda n: scores[n]["average_precision"])
    return models, winner, thresholds, scores


def explain(model, background, frame):
    """Explain the complete selected pipeline in probability units, including preprocessing."""
    import shap
    background = shap.sample(background[FEATURES], min(40, len(background)), random_state=42)
    def predict(values):
        return model.predict_proba(pd.DataFrame(values, columns=FEATURES))[:, 1]
    explainer = shap.Explainer(predict, shap.maskers.Independent(background), algorithm="permutation",
                               feature_names=FEATURES, seed=42)
    return explainer(frame[FEATURES], max_evals=2*len(FEATURES)+1)


def predict(model, features):
    missing = set(FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return model.predict_proba(features[FEATURES])[:, 1]
