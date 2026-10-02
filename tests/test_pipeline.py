import numpy as np
import pandas as pd
from src.data import build_dataset, load_events, FEATURES
from src.model import chronological_split, fit_models, predict, explain


def prices():
    dates = pd.bdate_range("2020-01-01", periods=180)
    return pd.DataFrame({"Close": np.arange(180) + 100., "Volume": 1000.}, index=dates)


def test_features_ignore_future_prices_and_label_uses_five_sessions(tmp_path):
    event_path = tmp_path / "events.csv"
    pd.DataFrame({"Ticker": ["TEST"], "Date Added": ["2020-05-01"],
                  "# Laid Off": [None], "%": ["10%"]}).to_csv(event_path, index=False)
    events = load_events(event_path)
    stock = prices(); market = prices()
    original = build_dataset(events, {"TEST": stock}, market, as_of="2021-01-01")
    prediction_date = original.prediction_date.iloc[0]
    mutated = stock.copy(); mutated.loc[mutated.index >= prediction_date, "Close"] *= 2
    altered = build_dataset(events, {"TEST": mutated}, market, as_of="2021-01-01")
    pd.testing.assert_frame_equal(original[FEATURES], altered[FEATURES])
    assert original.target_end.iloc[0] == stock.index[stock.index.get_loc(prediction_date)+4]
    assert original["count_missing"].iloc[0] == 1
    immature = build_dataset(events, {"TEST": stock}, market, as_of="2020-05-06")
    assert pd.isna(immature.target.iloc[0])


def synthetic_dataset():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=150)
    data = pd.DataFrame(rng.normal(size=(150, len(FEATURES))), columns=FEATURES)
    data["prediction_date"] = dates
    data["target_end"] = dates + pd.offsets.BDay(4)
    data["target"] = np.arange(150) % 2
    data.loc[::7, "layoff_count"] = np.nan
    return data


def test_split_purges_overlap_and_keeps_same_dates_together():
    data = pd.concat([synthetic_dataset()]*2, ignore_index=True)
    train, val, test = chronological_split(data)
    assert train.target_end.max() < val.prediction_date.min()
    assert val.target_end.max() < test.prediction_date.min()
    assert not set(train.prediction_date) & set(val.prediction_date)


def test_models_predictions_and_shap_add_up():
    train, val, test = chronological_split(synthetic_dataset())
    models, winner, thresholds, scores = fit_models(train, val)
    assert set(models) == {"baseline", "logistic", "xgboost"}
    for model in models.values():
        probabilities = predict(model, test)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
    # Exercise explanations for XGBoost regardless of which model wins.
    frame = test.iloc[:2]
    explanation = explain(models["xgboost"], train, frame)
    np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1),
                               predict(models["xgboost"], frame), atol=1e-6)


def test_offline_pipeline_artifacts_and_dashboard(tmp_path, monkeypatch):
    import argparse
    import json
    from pathlib import Path
    from src import pipeline
    from streamlit.testing.v1 import AppTest
    dates = pd.bdate_range("2019-01-01", periods=450)
    rng = np.random.default_rng(4)
    cache = tmp_path / "data/prices"
    cache.mkdir(parents=True)
    for ticker in ["TEST", "SPY"]:
        pd.DataFrame({"Close": 100*np.exp(np.cumsum(rng.normal(0, .02, len(dates)))),
                      "Volume": rng.integers(100, 1000, len(dates))},
                     index=pd.Index(dates, name="Date")).to_csv(cache / f"{ticker}.csv")
    event_file = tmp_path / "events.csv"
    pd.DataFrame({"Ticker": "TEST", "Date Added": dates[80:400:2],
                  "# Laid Off": 100, "%": "10%"}).to_csv(event_file, index=False)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    pipeline.run(argparse.Namespace(events=str(event_file), output=str(tmp_path / "artifacts"),
                 as_of="2021-01-01", benchmark="SPY", horizon=5, refresh=False, offline=True))
    metadata = json.loads((tmp_path / "artifacts/metadata.json").read_text())
    assert metadata["test"][metadata["model"]]["n"] > 0
    assert (tmp_path / "artifacts/drift.csv").exists()
    assert (tmp_path / "artifacts/run_history.csv").exists()
    source = Path("app.py").read_text().replace('ARTIFACTS = ROOT / "artifacts"',
                  f'ARTIFACTS = Path({str(tmp_path / "artifacts")!r})')
    app = AppTest.from_string(source).run(timeout=90)
    assert not app.exception, str(app.exception)
    app.slider[0].set_value(.7).run(timeout=90)
    assert not app.exception, str(app.exception)

    app.button[1].click().run(timeout=90)
    assert not app.exception, str(app.exception)
    app.multiselect[0].set_value([]).run(timeout=90)
    assert not app.exception, str(app.exception)
