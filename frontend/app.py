"""
Frontend - Streamlit dashboard for the NIDS mini project (Stage 4).

All predictions come from backend/predictor.py, which loads
resources/artifacts/nids_model.joblib (preprocessor + model in one Pipeline, saved by
ml/classical_models.py). This file only builds the page.

Tabs:
  Analyse        pick a held-out test connection, run an animated scan, compare with the truth
  Simulator      change feature values with sliders and watch the speed dial react live
  Live replay    stream test connections one by one, like a monitoring console
  Batch upload   predict a whole CSV file
  Model insights results of the deployed logistic regression: scores on two test sets,
                 confusion matrices, ROC / precision-recall curves, score distribution,
                 feature ranking, dataset balance

Visual helpers (CSS, animated dial, event feed) live in frontend/ui.py.

Run from the project root:
    streamlit run frontend/app.py
"""

import random
import sys
import time
from pathlib import Path

# make the project root importable (backend/, ml/) when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend import predictor  # noqa: E402
from frontend import ui  # noqa: E402

DATASET_NAMES = {"nslkdd": "NSL-KDD", "cicids2017": "CICIDS2017"}

st.set_page_config(page_title="ML Network Intrusion Detection", page_icon="🛡️", layout="wide")
ui.inject_css()


# ---------------------------------------------------------------------------
# Cached access to the backend (load files once, not on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_bundle():
    return predictor.load_bundle()


@st.cache_data
def load_csv(name):
    return predictor.load_csv(name)


@st.cache_data
def load_json(name):
    return predictor.load_json(name)


def dial_text(bundle, prediction):
    """Verdict and small print shown under the dial."""
    is_attack = prediction != "Normal"
    verdict = "ATTACK DETECTED" if is_attack else "NORMAL TRAFFIC"
    if len(bundle["class_names"]) == 2:
        detail = "Flagged as attack when the probability is 50% or more"
    else:
        detail = f"Predicted class: {prediction}"
    return is_attack, verdict, detail


def pretty(name):
    return name.replace("_", " ").title()


def zscore_chart(df):
    df = df.assign(shown=df["z"].clip(-6, 6), size=df["z"].abs()).sort_values("size", ascending=False)
    return alt.Chart(df).mark_bar(cornerRadius=3).encode(
        x=alt.X("shown:Q", title="standard deviations from the training average (clipped at ±6)",
                scale=alt.Scale(domain=[-6, 6])),
        y=alt.Y("feature:N", sort=list(df["feature"]), title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        color=alt.condition(alt.datum.size > 2, alt.value("#f59e0b"), alt.value(ui.CYAN)),
        tooltip=["feature", alt.Tooltip("value:Q", format=",.4g"), alt.Tooltip("z:Q", format=".2f", title="z-score")],
    ).properties(height=max(240, 26 * len(df)))


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

if not predictor.model_available():
    ui.hero("Network Intrusion Detection", "No trained model found yet.", ["Model offline"])
    st.error("Run these first, from the project folder:")
    st.code("python -m ml.preprocessing --dataset nslkdd --data ml/data/nsl-kdd/KDDTrain+.txt\n"
            "python -m ml.classical_models", language="bash")
    st.stop()

bundle = load_bundle()
features = bundle["selected_features"]
categorical = set(bundle["categorical_features"])
binary = len(bundle["class_names"]) == 2
metrics = bundle["test_metrics"]
samples = load_csv("test_samples.csv")
defaults = predictor.training_defaults(bundle)

ui.hero(
    "Network Intrusion Detection System",
    "Machine-learning model that classifies network connections as <b>normal</b> or "
    "<b>attack</b> from their traffic features.",
    ["Model online", pretty(bundle["model_name"]), DATASET_NAMES[bundle["dataset"]],
     f"{len(features)} features", f"Test F1 {metrics['f1']:.4f}"])

with st.sidebar:
    st.markdown("### 🛡️ Model card")
    st.write(f"**{pretty(bundle['model_name'])}** trained on "
             f"**{DATASET_NAMES[bundle['dataset']]}** with {len(features)} selected features.")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("F1 score", f"{metrics['f1']:.4f}")
    c1.metric("Precision", f"{metrics['precision']:.2%}")
    c2.metric("Recall", f"{metrics['recall']:.2%}")
    if metrics.get("false_positive_rate") is not None:
        st.metric("False alarms (normal flagged)", f"{metrics['false_positive_rate']:.2%}")
    st.caption("Scores on the held-out 20% test set.")
    st.info("**Limits:** a supervised model recognises the attack types it was trained on. "
            "New (zero-day) attack types can be missed. Input is supplied flow features, "
            "not live packet capture.", icon="⚠️")

tab_analyse, tab_sim, tab_live, tab_batch, tab_insights = st.tabs(
    ["🎯 Analyse", "🎛️ Simulator", "📡 Live replay", "📂 Batch upload", "📊 Model insights"])


# ---------------------------------------------------------------------------
# 1. Analyse one held-out test connection
# ---------------------------------------------------------------------------

def pick_random_example():
    pool = samples.index if st.session_state.get("pool", "Any") == "Any" else \
        samples.index[samples["true_class"] == st.session_state["pool"]]
    st.session_state["example_idx"] = int(random.choice(list(pool)))


with tab_analyse:
    if samples is None:
        st.warning("resources/artifacts/test_samples.csv not found. Run Phase 1 to create it.")
    else:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            ui.section("1 · Choose a connection")
            pool = st.segmented_control("Pick from", ["Any"] + sorted(samples["true_class"].unique()),
                                        default="Any", key="pool")
            options = samples.index if pool in (None, "Any") else \
                samples.index[samples["true_class"] == pool]
            if st.session_state.get("example_idx") not in options:
                st.session_state["example_idx"] = int(options[0])
            c1, c2 = st.columns([3, 1])
            index = c1.selectbox("Connection from the held-out test set", options, key="example_idx",
                                 format_func=lambda i: f"Connection #{i:03d}")
            c2.button("🎲 Random", on_click=pick_random_example, width="stretch")

            row = samples.loc[[index]]
            table = row[features].T.rename(columns={index: "value"}).astype(str)
            st.dataframe(table, height=260, width="stretch")
            analyse = st.button("⚡ Analyse connection", type="primary", width="stretch")
            st.caption("The true label stays hidden until the model has made its prediction.")

        if analyse:  # store the result so later reruns redraw the same panel without re-animating
            result = predictor.predict(bundle, row).iloc[0]
            st.session_state["scan"] = {"idx": int(index), "value": float(result["attack_probability"]),
                                        "prediction": result["prediction"]}

        with right:
            ui.section("2 · Threat analysis")
            scan = st.session_state.get("scan")
            if scan and scan["idx"] == index:
                is_attack, verdict, detail = dial_text(bundle, scan["prediction"])
                ui.threat_dial(scan["value"], is_attack, verdict, detail, features,
                               calibrate=True, threshold=0.5 if binary else None)
                truth = samples.loc[index, "true_class"]
                if truth == scan["prediction"]:
                    st.success(f"True class: **{truth}**, the model is correct ✔")
                else:
                    st.error(f"True class: **{truth}**, the model is wrong ✘")
            else:
                ui.standby_panel("Press <b>⚡ Analyse connection</b> to scan this connection.")

        if scan and scan["idx"] == index:
            ui.section("How unusual is this connection?")
            st.altair_chart(zscore_chart(predictor.standardised_values(bundle, row)), width="stretch")
            st.markdown('<div class="note">Each bar shows how far a feature value is from the average '
                        'of the training data (orange = more than 2 standard deviations). It describes the '
                        'connection; it is not a measure of how much the model relied on that feature.</div>',
                        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. What-if simulator: live dial that follows the sliders
# ---------------------------------------------------------------------------

def set_inputs(values):
    for f in features:
        value = values[f]
        st.session_state[f"f_{f}"] = value if f in categorical else round(float(value), 4)


def typical_profile(true_class):
    """Median (numbers) / most common value (categories) of the demo rows of one class."""
    rows = samples[samples["true_class"] == true_class]
    return {f: rows[f].mode()[0] if f in categorical else rows[f].median() for f in features}


with tab_sim:
    if "f_" + features[0] not in st.session_state:
        set_inputs(defaults)

    ui.section("Presets")
    p1, p2, p3, _ = st.columns([1, 1, 1, 2])
    p1.button("↺ Training median", on_click=set_inputs, args=(defaults,), width="stretch")
    if samples is not None:
        p2.button("✅ Typical normal", on_click=set_inputs, args=(typical_profile("Normal"),),
                  width="stretch")
        attack_classes = [c for c in samples["true_class"].unique() if c != "Normal"]
        p3.button("🚨 Typical attack", on_click=set_inputs, args=(typical_profile(attack_classes[0]),),
                  width="stretch")
    st.caption("Presets use the median of the demo test rows of each class. "
               "Move any control and the dial reacts immediately.")

    controls, dial_col = st.columns([1.5, 1], gap="large")
    values = {}
    with controls:
        cat_feats = [f for f in features if f in categorical]
        rate_feats = [f for f in features if f not in categorical and f.endswith("_rate")]
        other_feats = [f for f in features if f not in categorical and f not in rate_feats]

        if cat_feats:
            ui.section("Connection type")
            cols = st.columns(len(cat_feats))
            for col, f in zip(cols, cat_feats):
                options = bundle["categories"][f]
                with col:
                    if len(options) <= 4:
                        values[f] = st.segmented_control(f, options, key=f"f_{f}")
                    else:
                        values[f] = st.selectbox(f, options, key=f"f_{f}")
                if values[f] is None:  # segmented control can be clicked off
                    values[f] = defaults[f]
        if rate_feats:
            ui.section("Rates (share of connections, 0-1)")
            cols = st.columns(2)
            for i, f in enumerate(rate_feats):
                values[f] = cols[i % 2].slider(f, 0.0, 1.0, step=0.01, key=f"f_{f}")
        if other_feats:
            ui.section("Counts and sizes")
            cols = st.columns(3)
            for i, f in enumerate(other_feats):
                values[f] = cols[i % 3].number_input(f, min_value=0.0, key=f"f_{f}", format="%g")

    with dial_col:
        result = predictor.predict(bundle, pd.DataFrame([values])).iloc[0]
        value = float(result["attack_probability"])
        is_attack, verdict, detail = dial_text(bundle, result["prediction"])
        ui.section("Live threat dial")
        ui.threat_dial(value, is_attack, verdict, detail, features,
                       previous=st.session_state.get("sim_previous", 0.0), calibrate=False,
                       threshold=0.5 if binary else None)
        st.session_state["sim_previous"] = value  # next change animates from here

    ui.section("What-if curve: sweep one feature, keep the others fixed")
    numeric_feats = [f for f in features if f not in categorical]
    c1, c2 = st.columns([1, 2])
    sweep = c1.selectbox("Feature to sweep", numeric_feats,
                         index=numeric_feats.index("serror_rate") if "serror_rate" in numeric_feats else 0)
    if sweep.endswith("_rate"):
        high = 1.0
    else:  # up to a bit above the largest value seen in the demo rows
        high = float(max(samples[sweep].quantile(0.99) * 1.2 if samples is not None else 1.0,
                         values[sweep] * 1.5, 1.0))
    grid = np.linspace(0.0, high, 60)
    curve_rows = pd.DataFrame([values] * len(grid))
    curve_rows[sweep] = grid
    curve = pd.DataFrame({sweep: grid, "probability": predictor.predict(bundle, curve_rows)["attack_probability"].to_numpy()})
    line = alt.Chart(curve).mark_area(
        line={"color": ui.CYAN}, interpolate="monotone",
        color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                           stops=[alt.GradientStop(color="rgba(34,211,238,0.02)", offset=0),
                                  alt.GradientStop(color="rgba(34,211,238,0.35)", offset=1)])).encode(
        x=alt.X(f"{sweep}:Q", title=sweep),
        y=alt.Y("probability:Q", title="attack probability", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip(f"{sweep}:Q", format=",.4g"), alt.Tooltip("probability:Q", format=".1%")])
    now = alt.Chart(pd.DataFrame({sweep: [values[sweep]]})).mark_rule(color="#f8fafc", strokeDash=[4, 3]).encode(
        x=f"{sweep}:Q")
    layers = [line, now]
    if binary:
        layers.append(alt.Chart(pd.DataFrame({"probability": [0.5]})).mark_rule(color=ui.RED, opacity=0.6).encode(
            y="probability:Q"))
    c2.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")
    c1.markdown('<div class="note">The curve shows the model\'s attack probability as the chosen '
                'feature changes, with every other control kept at its current value. The dashed '
                'white line is the current value; the red line is the 50% decision threshold. '
                'The curve is smooth because logistic regression turns a weighted sum into a probability.</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Live replay of test connections
# ---------------------------------------------------------------------------

def timeline_chart(history):
    df = pd.DataFrame(history)
    base = alt.Chart(df).encode(x=alt.X("n:Q", title="connection"))
    line = base.mark_line(color=ui.CYAN, strokeWidth=2, interpolate="monotone").encode(
        y=alt.Y("probability:Q", title="attack probability", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")))
    points = base.mark_circle(size=70, opacity=1).encode(
        y="probability:Q",
        color=alt.condition(alt.datum.prediction == "Normal", alt.value(ui.GREEN), alt.value(ui.RED)),
        tooltip=["n", "prediction", alt.Tooltip("probability:Q", format=".1%"), "true_class"])
    rule = alt.Chart(pd.DataFrame({"probability": [0.5]})).mark_rule(strokeDash=[4, 4], color="#94a3b8").encode(
        y="probability:Q")
    layers = [line, points] + ([rule] if binary else [])
    return alt.layer(*layers).properties(height=230)


with tab_live:
    if samples is None:
        st.warning("resources/artifacts/test_samples.csv not found. Run Phase 1 to create it.")
    else:
        st.markdown('<div class="note">Replay mode: connections from the held-out test set are fed '
                    'to the model one at a time, like a monitoring console. This is a replay of '
                    'recorded features, not live packet capture.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1.3])
        count = c1.slider("Connections to replay", 10, len(samples), min(30, len(samples)))
        speed = c2.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")
        start = c3.button("▶ Start monitoring", type="primary", width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        tiles = [m.empty() for m in (m1, m2, m3, m4)]
        chart_box = st.empty()
        ui.section("Event feed (newest first)")
        feed_box = st.empty()

        def draw(history):
            attacks = sum(h["prediction"] != "Normal" for h in history)
            correct = sum(h["correct"] for h in history)
            tiles[0].metric("Analysed", len(history))
            tiles[1].metric("🚨 Attacks flagged", attacks)
            tiles[2].metric("✅ Normal", len(history) - attacks)
            tiles[3].metric("Match true label", f"{correct / len(history):.0%}" if history else "–")
            if history:
                chart_box.altair_chart(timeline_chart(history), width="stretch")
                with feed_box.container():
                    ui.event_feed(list(reversed(history[-8:])))

        if start:
            stream = samples.sample(count).reset_index(drop=True)  # new random order every run
            results = predictor.predict(bundle, stream)  # predict all at once, then reveal one by one
            delay = {"Slow": 0.8, "Normal": 0.4, "Fast": 0.12}[speed]
            history = []
            for i in range(count):
                history.append({"n": i + 1, "prediction": results.loc[i, "prediction"],
                                "probability": float(results.loc[i, "attack_probability"]),
                                "true_class": stream.loc[i, "true_class"],
                                "correct": results.loc[i, "prediction"] == stream.loc[i, "true_class"]})
                draw(history)
                time.sleep(delay)
            st.session_state["replay"] = history
            attacks = sum(h["prediction"] != "Normal" for h in history)
            st.toast(f"Replay finished: {attacks} of {count} connections flagged as attacks", icon="📡")
        elif st.session_state.get("replay"):
            draw(st.session_state["replay"])  # keep the last replay visible
        else:
            draw([])
            feed_box.markdown('<div class="note">Press ▶ Start monitoring.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Batch prediction for a CSV file
# ---------------------------------------------------------------------------

with tab_batch:
    st.write(f"Upload a CSV with the {len(features)} selected feature columns "
             "(extra columns are ignored), or try the built-in demo file.")
    with st.expander("Required columns"):
        st.code(", ".join(features), language=None)
    c1, c2 = st.columns([3, 1])
    uploaded = c1.file_uploader("CSV file", type="csv")
    if c2.button("Use demo file", width="stretch"):
        st.session_state["use_demo_file"] = True
    if uploaded is not None:
        st.session_state["use_demo_file"] = False

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif st.session_state.get("use_demo_file") and samples is not None:
        data = samples.copy()
        st.caption("Using resources/artifacts/test_samples.csv")

    if data is not None:
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results = predictor.predict(bundle, data)
            attacks = int((results["prediction"] != "Normal").sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Connections", len(results))
            m2.metric("🚨 Attacks", attacks)
            m3.metric("✅ Normal", len(results) - attacks)
            if "true_class" in data.columns:
                m4.metric("Accuracy vs true_class", f"{(results['prediction'] == data['true_class']).mean():.1%}")

            chart1, chart2 = st.columns(2)
            counts = results["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
            donut = alt.Chart(counts).mark_arc(innerRadius=60, cornerRadius=4).encode(
                theta="count:Q",
                color=alt.Color("prediction:N", scale=alt.Scale(
                    domain=list(counts["prediction"]),
                    range=[ui.GREEN if p == "Normal" else ui.RED for p in counts["prediction"]])),
                tooltip=["prediction", "count"]).properties(height=240, title="Predictions")
            chart1.altair_chart(donut, width="stretch")
            hist = alt.Chart(results).mark_bar(color=ui.CYAN, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("attack_probability:Q", bin=alt.Bin(maxbins=20), title="attack probability"),
                y=alt.Y("count()", title="connections")).properties(height=240, title="How confident is the model?")
            chart2.altair_chart(hist, width="stretch")

            table = pd.concat([results, data], axis=1)
            table.insert(0, "status", np.where(table["prediction"] == "Normal", "✅", "🚨"))
            st.dataframe(table, height=380, width="stretch", column_config={
                "attack_probability": st.column_config.ProgressColumn(
                    "attack probability", min_value=0.0, max_value=1.0, format="percent")})
            st.download_button("⬇ Download predictions (CSV)", table.to_csv(index=False),
                               "nids_predictions.csv", "text/csv")


# ---------------------------------------------------------------------------
# 5. Model insights (results of stages 1-3)
# ---------------------------------------------------------------------------

with tab_insights:
    model_name = bundle["model_name"]
    model_label = pretty(model_name)
    result = load_json(f"results/{model_name}.json")
    curves = load_json("curves.json")
    curve = curves.get(model_name) if curves else None
    metadata = load_json("metadata.json")
    sub_perf, sub_curves, sub_features, sub_data = st.tabs(
        ["📈 Performance", "🎯 ROC & PR curves", "🧠 Features", "🗂️ Dataset"])

    def score_tiles(scores):
        cols = st.columns(5, gap="small")
        cols[0].metric("Accuracy", f"{scores['accuracy']:.2%}")
        cols[1].metric("Precision", f"{scores['precision']:.2%}")
        cols[2].metric("Recall", f"{scores['recall']:.2%}")
        cols[3].metric("F1 score", f"{scores['f1']:.4f}")
        if scores.get("false_positive_rate") is not None:
            cols[4].metric("False alarms", f"{scores['false_positive_rate']:.2%}")

    def confusion_chart(cm, title):
        names = bundle["class_names"]
        cm = np.array(cm)
        cells = pd.DataFrame([(names[i], names[j], int(cm[i, j]))
                              for i in range(len(names)) for j in range(len(names))],
                             columns=["actual", "predicted", "count"])
        base = alt.Chart(cells).encode(x=alt.X("predicted:N", sort=names), y=alt.Y("actual:N", sort=names))
        heat = base.mark_rect(cornerRadius=8).encode(
            color=alt.Color("count:Q", scale=alt.Scale(type="symlog", scheme="teals"), legend=None),
            tooltip=["actual", "predicted", "count"])
        text = base.mark_text(fontSize=20, fontWeight="bold").encode(
            text="count:Q",
            color=alt.condition(alt.datum.count > cm.max() / 3, alt.value("#0f172a"), alt.value("#e2e8f0")))
        return (heat + text).properties(height=260, title=title)

    # --- scores on both test sets, confusion matrices -------------------------------
    with sub_perf:
        if result is None:
            st.info(f"No saved results for {model_label}. Run `python -m ml.classical_models`.")
        else:
            extra = result.get("extra_test")
            ui.section(f"{model_label}: held-out 20% test split")
            score_tiles(result)
            if extra:
                ui.section("KDDTest+: separate file that includes attack types never seen in training")
                score_tiles(extra)
                if extra.get("unseen_attack_detection_rate") is not None:
                    st.metric("Attacks of unseen types that were detected",
                              f"{extra['unseen_attack_detection_rate']:.1%}")

                ui.section("The same model on the two test sets")
                rows = [{"test set": label, "metric": k.capitalize() if k != "f1" else "F1", "score": sc[k]}
                        for label, sc in [("Random 20% split", result), ("KDDTest+", extra)]
                        for k in ["accuracy", "precision", "recall", "f1"]]
                chart = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0), sort=["Accuracy", "Precision", "Recall", "F1"]),
                    xOffset="test set:N",
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                    color=alt.Color("test set:N", scale=alt.Scale(range=["#22d3ee", "#f59e0b"]),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=["test set", "metric", alt.Tooltip("score:Q", format=".2%")]).properties(height=300)
                st.altair_chart(chart, width="stretch")
                st.markdown('<div class="note">On the harder file precision stays high but recall falls: '
                            'attacks that look unlike the training data are labelled normal. A supervised '
                            'model cannot be relied on to catch zero-day attacks.</div>', unsafe_allow_html=True)

            ui.section("Confusion matrices")
            c1, c2 = st.columns(2)
            c1.altair_chart(confusion_chart(result["confusion_matrix"], "Random 20% split"), width="stretch")
            if extra:
                c2.altair_chart(confusion_chart(extra["confusion_matrix"], "KDDTest+"), width="stretch")
            st.markdown('<div class="note">Rows are the true classes, columns the predictions. Off-diagonal '
                        'cells are mistakes: top-right = false alarms (normal traffic flagged), '
                        'bottom-left = missed attacks.</div>', unsafe_allow_html=True)

    # --- ROC, precision-recall, score distribution ------------------------------------
    with sub_curves:
        if not curve:
            st.info("No curves found. Run `python -m ml.evaluation --curves` (binary task only).")
        else:
            m1, m2, _ = st.columns([1, 1, 2])
            m1.metric("ROC AUC", f"{curve['roc_auc']:.4f}")
            m2.metric("Average precision", f"{curve['average_precision']:.4f}")
            zoom = st.toggle("Zoom into the top-left corner", value=True)
            roc = pd.DataFrame({"fpr": curve["roc"]["fpr"], "tpr": curve["roc"]["tpr"]})
            pr = pd.DataFrame({"recall": curve["pr"]["recall"], "precision": curve["pr"]["precision"]})
            c1, c2 = st.columns(2)
            with c1:
                ui.section("ROC curve")
                x_dom, y_dom = ([0, 0.1], [0.8, 1.0]) if zoom else ([0, 1], [0, 1])
                line = alt.Chart(roc).mark_line(strokeWidth=2.5, color=ui.CYAN, clip=True).encode(
                    x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=x_dom)),
                    y=alt.Y("tpr:Q", title="true positive rate (recall)", scale=alt.Scale(domain=y_dom)),
                    tooltip=[alt.Tooltip("fpr:Q", format=".4f"), alt.Tooltip("tpr:Q", format=".4f")])
                guess = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(
                    strokeDash=[4, 4], color="#475569", clip=True).encode(
                    x=alt.X("fpr:Q", scale=alt.Scale(domain=x_dom)), y=alt.Y("tpr:Q", scale=alt.Scale(domain=y_dom)))
                layers = [guess, line]
                if result and result.get("false_positive_rate") is not None:  # the 0.5 threshold used by the app
                    point = pd.DataFrame({"fpr": [result["false_positive_rate"]], "tpr": [result["recall"]]})
                    layers.append(alt.Chart(point).mark_point(size=120, filled=True, color=ui.RED, clip=True).encode(
                        x="fpr:Q", y="tpr:Q", tooltip=[alt.Tooltip("fpr:Q", format=".4f"), alt.Tooltip("tpr:Q", format=".4f")]))
                st.altair_chart(alt.layer(*layers).properties(height=340).interactive(), width="stretch")
                st.caption("Red dot: the 50% decision threshold the app uses. Dashed line: random guessing.")
            with c2:
                ui.section("Precision-recall curve")
                dom = [0.8, 1.0] if zoom else [0, 1]
                line = alt.Chart(pr).mark_line(strokeWidth=2.5, color=ui.CYAN, clip=True).encode(
                    x=alt.X("recall:Q", scale=alt.Scale(domain=dom)), y=alt.Y("precision:Q", scale=alt.Scale(domain=dom)),
                    tooltip=[alt.Tooltip("recall:Q", format=".4f"), alt.Tooltip("precision:Q", format=".4f")])
                layers = [line]
                if result:
                    point = pd.DataFrame({"recall": [result["recall"]], "precision": [result["precision"]]})
                    layers.append(alt.Chart(point).mark_point(size=120, filled=True, color=ui.RED, clip=True).encode(
                        x="recall:Q", y="precision:Q",
                        tooltip=[alt.Tooltip("recall:Q", format=".4f"), alt.Tooltip("precision:Q", format=".4f")]))
                st.altair_chart(alt.layer(*layers).properties(height=340).interactive(), width="stretch")
                st.caption("Scroll to zoom and drag to pan the charts.")

            if "histogram" in curve:
                ui.section("Score distribution: how well does the model separate the classes?")
                c1, c2 = st.columns([1, 3])
                log_scale = c1.toggle("Log scale", value=True)
                c1.markdown('<div class="note">Attack probability given to every test connection, split by its '
                            'true class. Good separation = normal traffic piled up on the left, attacks on the right.</div>',
                            unsafe_allow_html=True)
                h = curve["histogram"]
                hist = pd.concat([pd.DataFrame({"bin": h["bin_start"], "count": counts, "true class": cls})
                                  for cls, counts in h["counts"].items()])
                if log_scale:  # a log axis cannot show 0, so empty bins are left out
                    hist = hist[hist["count"] > 0]
                chart = alt.Chart(hist).mark_bar(opacity=0.75, binSpacing=1).encode(
                    x=alt.X("bin:Q", bin=alt.Bin(step=0.05, extent=[0, 1]), title="attack probability",
                            axis=alt.Axis(format="%")),
                    y=alt.Y("count:Q", stack=None, title="test connections",
                            scale=alt.Scale(type="log" if log_scale else "linear")),
                    color=alt.Color("true class:N", scale=alt.Scale(domain=["Normal", "Attack"], range=[ui.GREEN, ui.RED])),
                    tooltip=["true class", alt.Tooltip("bin:Q", format=".0%", title="from"), "count"]).properties(height=300)
                c2.altair_chart(chart, width="stretch")

    # --- feature importance ------------------------------------------------------
    with sub_features:
        importance = load_csv("feature_importance.csv")
        if importance is not None:
            ui.section("Feature ranking used to pick the 20 inputs (Phase 1)")
            n_show = st.slider("Features to show", 10, len(importance), min(len(features) + 5, len(importance)))
            top = importance.head(n_show).copy()
            top["selected"] = np.where(top["feature"].isin(features), "selected", "not selected")
            chart = alt.Chart(top).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                x=alt.X("importance:Q", title="importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("selected:N", scale=alt.Scale(domain=["selected", "not selected"],
                                                              range=["#22d3ee", "#334155"])),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")]).properties(height=22 * n_show + 40)
            st.altair_chart(chart, width="stretch")
            cumulative = importance.assign(rank=np.arange(1, len(importance) + 1),
                                           cumulative=importance["importance"].cumsum())
            ui.section("Cumulative importance: why keep only the top features?")
            cum = alt.Chart(cumulative).mark_line(point=True, color=ui.CYAN).encode(
                x=alt.X("rank:Q", title="number of features kept"),
                y=alt.Y("cumulative:Q", title="share of total importance", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                tooltip=["rank", "feature", alt.Tooltip("cumulative:Q", format=".1%")])
            cut = alt.Chart(pd.DataFrame({"rank": [len(features)]})).mark_rule(color="#f59e0b", strokeDash=[4, 3]).encode(
                x="rank:Q")
            st.altair_chart((cum + cut).properties(height=260), width="stretch")
            kept = cumulative.loc[len(features) - 1, "cumulative"]
            st.caption(f"The {len(features)} selected features (orange line) carry {kept:.1%} of the "
                       f"total importance. The ranking was computed once during feature selection with a "
                       f"Random Forest helper; it is not the deployed model.")

    # --- dataset ----------------------------------------------------------------------
    with sub_data:
        if metadata:
            ui.section(f"Dataset: {DATASET_NAMES.get(metadata['dataset'], metadata['dataset'])} "
                       f"({metadata['task']} task)")
            counts = pd.concat([
                pd.DataFrame({"class": list(metadata["train_counts_before_balancing"]),
                              "rows": list(metadata["train_counts_before_balancing"].values()), "split": "train"}),
                pd.DataFrame({"class": list(metadata["test_counts"]),
                              "rows": list(metadata["test_counts"].values()), "split": "test"})])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Training rows", f"{metadata['train_rows']:,}")
            m2.metric("Test rows", f"{metadata['test_rows']:,}")
            m3.metric("Selected features", len(metadata["selected_features"]))
            m4.metric("Imbalance handling", metadata["balance_method"].replace("_", " "))
            c1, c2 = st.columns(2)
            split_chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("split:N", title=None, sort=["train", "test"]), xOffset="class:N",
                y=alt.Y("rows:Q", title="rows"),
                color=alt.Color("class:N", scale=alt.Scale(domain=list(bundle["class_names"]),
                                                           range=[ui.GREEN, ui.RED, "#f59e0b", "#a78bfa", "#22d3ee",
                                                                  "#f472b6", "#eab308", "#64748b"][:len(bundle["class_names"])])),
                tooltip=["split", "class", alt.Tooltip("rows:Q", format=",")]).properties(height=300, title="Rows per class")
            c1.altair_chart(split_chart, width="stretch")
            train_counts = counts[counts["split"] == "train"]
            donut = alt.Chart(train_counts).mark_arc(innerRadius=70, cornerRadius=4).encode(
                theta="rows:Q",
                color=alt.Color("class:N", scale=alt.Scale(domain=list(bundle["class_names"]),
                                                           range=[ui.GREEN, ui.RED, "#f59e0b", "#a78bfa", "#22d3ee",
                                                                  "#f472b6", "#eab308", "#64748b"][:len(bundle["class_names"])])),
                tooltip=["class", alt.Tooltip("rows:Q", format=",")]).properties(height=300, title="Training class share")
            c2.altair_chart(donut, width="stretch")
            st.caption("The test split keeps the real class mix. Balancing (class weights or SMOTE) "
                       "is applied to the training split only.")
