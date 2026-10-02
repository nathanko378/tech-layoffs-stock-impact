"""Streamlit interface for saved Stock Impact Predictor artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from src.data import ROOT, FEATURES
from src.model import metrics, predict, explain

st.set_page_config(page_title="Stock Impact Predictor", page_icon="📈", layout="wide")
st.title("Stock Impact Predictor")
st.markdown(
    "**How might a stock perform after a layoff announcement?** "
    "Explore a model's estimate of whether it will outperform the market, "
    "see what shaped the prediction, and check how well the model performed on past events."
)
ARTIFACTS = ROOT / "artifacts"
required = ["model.joblib", "metadata.json", "test_predictions.csv", "dataset.csv", "shap_values.npz", "background.csv"]
if any(not (ARTIFACTS / name).exists() for name in required):
    st.info("Train the pipeline to generate model predictions, evaluation results, and SHAP explanations.")
    st.code("python -m src.pipeline\nstreamlit run app.py", language="bash")
    st.markdown("The pipeline downloads adjusted prices, builds five-session benchmark-relative labels, and evaluates on the latest held-out events. No sample scores are shown before training.")
    st.stop()

@st.cache_resource
def load_model(version):
    return joblib.load(ARTIFACTS / "model.joblib")

@st.cache_data
def load_tables(version):
    test = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["event_date", "prediction_date", "target_end"])
    data = pd.read_csv(ARTIFACTS / "dataset.csv", parse_dates=["event_date", "prediction_date", "target_end"])
    with np.load(ARTIFACTS / "shap_values.npz") as saved:
        explanations = {key: saved[key] for key in saved.files}
    return test, data, explanations

meta = json.loads((ARTIFACTS / "metadata.json").read_text())
version = meta["created_at"]
model = load_model(version)
test, data, saved = load_tables(version)
MODEL_NAMES = {"logistic": "Logistic regression", "xgboost": "XGBoost", "baseline": "Baseline"}
model_name = MODEL_NAMES.get(meta["model"], meta["model"])
benchmark_name = "SPY (S&P 500 ETF)" if meta["benchmark"] == "SPY" else meta["benchmark"]
with st.sidebar:
    st.subheader("Your guide")
    st.markdown("**Start here** · Understand the prediction\n\n"
                "**Predict** · Enter an announcement or try a scenario\n\n"
                "**Explain** · See which inputs matter\n\n"
                "**Performance** · Check the model's track record\n\n"
                "**Historical predictions** · Explore past results")
    st.divider()
    st.caption("CURRENT MODEL")
    st.write(model_name)
    st.caption(f"Benchmark: {benchmark_name}\n\nForecast window: {meta['horizon']} trading sessions\n\nPrice data before: {meta['as_of']}")
    st.divider()
    st.caption("Built by Nathan Ko · Research project")

tabs = st.tabs(["Start here", "Predict", "Explain", "Performance", "Historical predictions", "Methodology"])

with tabs[0]:
    st.subheader("A layoff announcement. A market prediction. An explanation.")
    st.write(
        f"This application uses layoff details and recent stock-market conditions to estimate "
        f"the probability that a company’s stock will outperform {benchmark_name} over the next "
        f"{meta['horizon']} trading sessions. It also lets you inspect the prediction and evaluate "
        "the model against outcomes it did not train on."
    )
    with st.container(border=True):
        st.markdown("#### What does ‘outperform’ mean?")
        st.write("The stock earns a higher percentage return than the benchmark over the same period. "
                 "Both can rise or fall: if the stock falls 1% and the benchmark falls 3%, the stock still outperforms.")
        st.info("A prediction of 65% is the model’s estimated probability of outperformance. "
                "It does not mean the stock is expected to rise 65%.")
    st.markdown("### Start with these three steps")
    steps = st.columns(3)
    with steps[0], st.container(border=True):
        st.markdown("#### 1 · Make a prediction")
        st.write("Open **Predict** and enter a ticker, announcement date, and layoff size. "
                 "The app loads prior price history for you. Leave unknown layoff details blank.")
        st.caption("Just exploring? Choose a prepared event below the new-announcement form.")
    with steps[1], st.container(border=True):
        st.markdown("#### 2 · Understand the result")
        st.write("Open **Explain** to see which inputs matter most across past predictions. "
                 "Choose an event to see how each input moves its estimated probability up or down.")
        st.caption("These SHAP charts explain the model’s reasoning; they do not establish cause and effect.")
    with steps[2], st.container(border=True):
        st.markdown("#### 3 · Check the evidence")
        st.write("Open **Performance** to compare the model with a simple baseline. "
                 "Use **Historical predictions** to compare past forecasts with observed outcomes.")
        st.caption("Performance is measured on later events held out from model training.")
    st.markdown("### How much confidence should you place in it?")
    evidence = meta["test"][meta["model"]]
    cols = st.columns(3)
    cols[0].metric("Events used for model evaluation", evidence["n"])
    cols[1].metric("Held-out ROC-AUC", f"{evidence['roc_auc']:.3f}" if evidence["roc_auc"] is not None else "Undefined",
                   help="Measures how well probabilities rank outperforming stocks above other stocks. 0.5 is chance-level ranking; 1.0 is perfect ranking.")
    cols[2].metric("Historical events with outcomes", meta["labeled"])
    auc = evidence["roc_auc"]
    if auc is not None and auc < .6:
        st.info("The current model has limited ability to distinguish outperforming stocks from other stocks. "
                "Use the dashboard to explore the experiment, and inspect the performance curves before relying on a prediction.")
    else:
        st.caption("Historical scores are evidence about this dataset, not a guarantee of future performance. "
                   "Check calibration and the uncertainty interval in Performance.")
    with st.expander("What goes into a prediction?"):
        st.write("Layoff count and workforce percentage; recent stock returns, volatility and trading volume; "
                 "recent benchmark conditions; and the company’s previous layoff events in this dataset. "
                 "Missing layoff values are handled by the trained model.")
        st.write("The forecast uses completed price sessions available before the prediction. "
                 "Historical announcement timing is approximated from date-only records, and some company/ticker matches require review.")
    st.caption("Research scope: this model predicts relative returns under a date-only timing convention. "
               "It does not measure whether layoffs caused a stock move. Details are in Methodology.")

with tabs[1]:
    st.subheader("Predict a new announcement")
    st.write(f"Enter a company’s layoff announcement to estimate its probability of outperforming "
             f"{benchmark_name} over {meta['horizon']} trading sessions. Price history is loaded automatically.")
    st.caption("Use a US-style stock ticker such as AAPL or MSFT. An internet connection is required for price retrieval.")
    with st.form("new_event"):
        company_col, date_col = st.columns(2)
        with company_col:
            ticker = st.text_input("Stock ticker", "AAPL", help="The company’s stock-market symbol, e.g. AAPL for Apple.")
        with date_col:
            event_date = st.date_input("Announcement date", help="Use the date the layoff was announced. Future dates are not supported.")
        count_col, percent_col = st.columns(2)
        with count_col:
            count = st.text_input("Employees laid off", placeholder="e.g. 500", help="Leave blank if the number is unknown.")
        with percent_col:
            percent = st.text_input("Workforce laid off (%)", placeholder="e.g. 2", help="Enter 2 for 2%. Leave blank if unknown.")
        new_submit = st.form_submit_button("Get prediction", type="primary")
    if new_submit:
        try:
            from src.predict import predict_event
            with st.spinner("Loading prior market data…"):
                probability, new_frame = predict_event(ticker, event_date,
                    float(count) if count.strip() else None, float(percent) if percent.strip() else None)
            st.metric("New event outperformance probability", f"{probability:.1%}")
            with st.expander("View the inputs used for this prediction"):
                st.dataframe(new_frame, use_container_width=True)
            with st.spinner("Explaining prediction…"):
                import shap
                explanation = explain(model, pd.read_csv(ARTIFACTS / "background.csv"), new_frame)
                shap.plots.waterfall(explanation[0], show=False)
                st.pyplot(plt.gcf()); plt.close("all")
        except Exception as exc:
            st.error(f"Could not predict this event: {exc}")
    st.divider()
    st.subheader("Explore a prepared event")
    st.write("Select a prepared event to inspect its prediction or edit its prior-market features for a scenario. Changing inputs creates a hypothetical scenario; it does not download live market data.")
    selected = st.selectbox("Prepared event", data.event_id.tolist())
    row = data.loc[data.event_id == selected].iloc[0]
    inputs = {}
    with st.expander("Advanced: edit inputs and test a scenario"), st.form("prediction"):
        cols = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                initial = "" if pd.isna(row[feature]) else str(float(row[feature]))
                inputs[feature] = st.text_input(feature.replace("_", " "), initial, help="Blank means missing. Returns and volatility use decimal fractions, e.g. 0.05 = 5%.")
        submit = st.form_submit_button("Predict scenario")
    if submit:
        try:
            frame = pd.DataFrame([{k: float(v) if v.strip() else np.nan for k, v in inputs.items()}])
            if np.isinf(frame.to_numpy()).any():
                raise ValueError("Inputs must be finite or blank.")
            if not pd.isna(frame.layoff_count.iloc[0]) and frame.layoff_count.iloc[0] < 0:
                raise ValueError("Layoff count must be nonnegative.")
            if not pd.isna(frame.layoff_percent.iloc[0]) and not 0 <= frame.layoff_percent.iloc[0] <= 100:
                raise ValueError("Layoff percentage must be between 0 and 100.")
            frame["count_missing"] = frame.layoff_count.isna().astype(int)
            frame["percent_missing"] = frame.layoff_percent.isna().astype(int)
            probability = predict(model, frame)[0]
            st.metric(f"Probability of outperforming {meta['benchmark']}", f"{probability:.1%}")
            st.write(f"Classification: {'outperform' if probability >= meta['threshold'] else 'not outperform'} at validation-selected threshold {meta['threshold']:.2f}.")
            with st.spinner("Explaining this prediction…"):
                import shap
                background = pd.read_csv(ARTIFACTS / "background.csv")
                explanation = explain(model, background, frame)
                shap.plots.waterfall(explanation[0], show=False)
                st.pyplot(plt.gcf()); plt.close("all")
        except ValueError as exc:
            st.error(str(exc))
    else:
        probability = predict(model, pd.DataFrame([row]))[0]
        st.metric("Prepared event probability", f"{probability:.1%}")
    st.caption("Prepared training-period events are demonstrations of inference, not held-out performance. Predicted probabilities may be poorly calibrated; inspect the Performance tab.")

with tabs[2]:
    import shap
    st.subheader("What shaped the predictions?")
    st.write("SHAP assigns each input a contribution to the prediction. Larger contributions mean "
             "a stronger effect on the model’s output. Start with the overall charts, then select a past event below.")
    st.caption("Contributions explain the selected model's probability of outperformance. They describe model behavior, not causal effects. Missing features are imputed inside the pipeline.")
    explanation = shap.Explanation(values=saved["values"], base_values=saved["base_values"],
                                   data=saved["data"], feature_names=FEATURES)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Which inputs matter most?")
        st.caption("Longer bars mean a larger average influence across held-out events.")
        shap.plots.bar(explanation, show=False)
        st.pyplot(plt.gcf()); plt.close("all")
    with b:
        st.markdown("#### How do inputs move predictions?")
        st.caption("Each dot is an event. Right increases the probability; left decreases it. Color shows the input’s value.")
        shap.plots.beeswarm(explanation, show=False)
        st.pyplot(plt.gcf()); plt.close("all")
    st.divider()
    st.markdown("#### Follow one prediction from its starting point to its result")
    st.caption("The waterfall starts at the background probability, then adds each input’s contribution.")
    idx = st.selectbox("Explain held-out event", range(len(test)), format_func=lambda i: test.iloc[i].event_id)
    st.metric("Predicted outperformance probability", f"{test.iloc[idx].probability:.1%}")
    shap.plots.waterfall(explanation[idx], show=False)
    st.pyplot(plt.gcf()); plt.close("all")

with tabs[3]:
    st.subheader("How well does the model work?")
    st.write("These results compare predictions with observed returns on events the model did not train on. "
             "Compare the selected model with the baseline, which gives every event the same probability.")
    with st.expander("How to read the metrics"):
        st.markdown("**ROC-AUC** measures ranking quality: 0.5 is chance-level, 1.0 is perfect.\n\n"
                    "**Average precision** summarizes the precision–recall curve; compare it with the positive-class prevalence.\n\n"
                    "**Precision** asks: of events predicted to outperform, how many did?\n\n"
                    "**Recall** asks: of events that outperformed, how many did the model identify?\n\n"
                    "**F1** balances precision and recall. **Calibration** checks whether stated probabilities match observed frequencies.")
    st.caption(f"Test period: {meta['splits']['test']['start']} to {meta['splits']['test']['end']}. Model and threshold were selected using validation events only.")
    threshold = st.slider("Explore classification threshold", 0.0, 1.0, float(meta["threshold"]), .01)
    score = metrics(test.target, test.probability, threshold)
    cols = st.columns(5)
    for col, key, label in zip(cols, ["roc_auc", "average_precision", "precision", "recall", "f1"],
                              ["ROC-AUC", "Average precision", "Precision", "Recall", "F1"]):
        col.metric(label, f"{score[key]:.3f}" if score[key] is not None else "Undefined")
    st.caption(f"N={score['n']} · positive prevalence={score['prevalence']:.1%} · Brier score={score['brier']:.3f} · selected threshold={meta['threshold']:.2f}")
    if meta["roc_auc_interval"]:
        low, high = meta["roc_auc_interval"]
        st.caption(f"Approximate 95% date-cluster bootstrap ROC-AUC interval: {low:.3f}–{high:.3f}. Small samples and repeated companies limit certainty.")
    comparison = pd.DataFrame(meta["test"]).T
    st.dataframe(comparison.drop(columns="confusion_matrix"), use_container_width=True)
    st.caption("Comparison uses each model's validation-selected threshold. The slider affects only the selected model's displayed metrics.")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        for name in meta["test"]:
            fpr, tpr, _ = roc_curve(test.target, test[f"probability_{name}"])
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        ax.legend(); st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        for name in meta["test"]:
            precision, recall, _ = precision_recall_curve(test.target, test[f"probability_{name}"])
            ax.plot(recall, precision, label=name)
        ax.axhline(score["prevalence"], linestyle="--", color="gray")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
        ax.legend(); st.pyplot(fig); plt.close(fig)
    left, right = st.columns(2)
    with left:
        st.write("Confusion matrix: rows = actual; columns = predicted")
        st.dataframe(pd.DataFrame(score["confusion_matrix"], index=["Not outperform", "Outperform"],
                                  columns=["Not outperform", "Outperform"]))
    with right:
        observed, predicted = calibration_curve(test.target, test.probability, n_bins=5, strategy="quantile")
        fig, ax = plt.subplots()
        ax.plot(predicted, observed, "o-"); ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration")
        st.pyplot(fig); plt.close(fig)

with tabs[4]:
    st.subheader("Compare past predictions with what happened")
    st.write("Filter the held-out events by ticker. Each point shows the forecast probability; "
             "its color indicates whether the stock actually outperformed. Excess return is the stock return minus the benchmark return.")
    tickers = st.multiselect("Held-out tickers", sorted(test.Ticker.unique()), default=sorted(test.Ticker.unique()))
    filtered = test[test.Ticker.isin(tickers)]
    if filtered.empty:
        st.info("No events match the selected tickers.")
    else:
        chart = alt.Chart(filtered).mark_circle(size=70).encode(
            x=alt.X("prediction_date:T", title="Prediction date"),
            y=alt.Y("probability:Q", title="Outperformance probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("target:N", title="Observed outperformance"),
            tooltip=["Ticker", "event_id", "probability", "excess_return"])
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(filtered[["event_id", "prediction_date", "target_end", "probability", "target", "excess_return"]], use_container_width=True)
        st.download_button("Download held-out predictions", filtered.to_csv(index=False), "predictions.csv", "text/csv")

with tabs[5]:
    st.subheader("How the experiment is built")
    st.write("The pipeline pairs announcements with adjusted stock and benchmark prices, builds features "
             "from prior sessions, and compares a baseline, logistic regression, and XGBoost. Earlier events "
             "train the models; later validation events select the model and threshold; the latest held-out events evaluate it.")
    st.warning(meta["date_caveat"] + ". Results describe this prediction convention, not a causal estimate of layoffs.")
    st.write(meta["target"])
    st.write(meta["timing"])
    st.dataframe(pd.DataFrame(meta["splits"]).T)
    st.write(f"Input events: {meta['events']} · usable: {meta['usable']} · mature labels: {meta['labeled']}")
    if meta["failures"]:
        st.warning(f"Price retrieval failed for {len(meta['failures'])} tickers. Review mappings and coverage before interpreting results.")
        st.json(meta["failures"])
    if (ARTIFACTS / "drift.csv").exists():
        st.subheader("Feature distribution monitoring")
        st.caption("Population stability index compares held-out features with training quantile bins. This is descriptive; small samples can produce large values. Missingness is reported separately.")
        st.dataframe(pd.read_csv(ARTIFACTS / "drift.csv"), use_container_width=True)
    if (ARTIFACTS / "run_history.csv").exists():
        st.subheader("Training run history")
        st.caption("Scores across runs may use different test periods and are not directly comparable.")
        st.dataframe(pd.read_csv(ARTIFACTS / "run_history.csv"), use_container_width=True)
    with st.expander("Technical model details and local retraining"):
        st.code("python -m src.pipeline --refresh", language="bash")
        st.json({k: meta[k] for k in ["created_at", "data_sha256", "versions", "validation"]})
