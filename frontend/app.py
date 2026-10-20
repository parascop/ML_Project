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
  Model insights model comparison, ROC / precision-recall curves, score distributions,
                 feature importance, confusion matrix, training curves, dataset balance

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
        line={"color": ui.CYAN}, interpolate="step-after",
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
                'Steps appear because tree models split features at fixed thresholds.</div>',
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
    comparison = load_csv("model_comparison.csv")
    curves = load_json("curves.json")
    metadata = load_json("metadata.json")
    if comparison is not None:
        comparison = comparison.assign(model=comparison["model"].map(pretty))
    sub_perf, sub_curves, sub_features, sub_training, sub_data = st.tabs(
        ["📈 Performance", "🎯 ROC & PR curves", "🧠 Features", "🔁 Training", "🗂️ Dataset"])

    # --- model comparison, harder test set, confusion matrix ------------------
    with sub_perf:
        if comparison is not None:
            ui.section("Model comparison (held-out test set)")
            long = comparison.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
                                   var_name="metric", value_name="score")
            bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("model:N", sort=list(comparison["model"]), title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="metric:N",
                y=alt.Y("score:Q", scale=alt.Scale(domain=[max(0, long["score"].min() - 0.02), 1]), title=None),
                color=alt.Color("metric:N", scale=alt.Scale(range=["#22d3ee", "#a78bfa", "#f59e0b", "#22c55e"])),
                tooltip=["model", "metric", alt.Tooltip("score:Q", format=".4f")]).properties(height=300)
            st.altair_chart(bars, width="stretch")
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.caption(f"Deployed in this app: {pretty(bundle['model_name'])} (highest F1 on the held-out test set).")

            if "extra_test_f1" in comparison.columns:
                ui.section("Harder test: KDDTest+ with attack types never seen in training")
                extra = comparison.melt(id_vars="model", value_vars=["extra_test_f1", "unseen_attack_detection"],
                                        var_name="measure", value_name="score")
                extra["measure"] = extra["measure"].map({"extra_test_f1": "F1 on KDDTest+",
                                                         "unseen_attack_detection": "unseen attack types detected"})
                chart = alt.Chart(extra).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="measure:N",
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                    color=alt.Color("measure:N", scale=alt.Scale(range=["#22d3ee", "#ef4444"])),
                    tooltip=["model", "measure", alt.Tooltip("score:Q", format=".1%")]).properties(height=260)
                st.altair_chart(chart, width="stretch")
                st.markdown('<div class="note">Every model detects only about a third of the attacks whose '
                            'type was not in the training data. A supervised model cannot be relied on to '
                            'catch zero-day attacks.</div>', unsafe_allow_html=True)

        results_json = load_json(f"results/{bundle['model_name']}.json")
        if results_json is not None:
            ui.section(f"Confusion matrix · {pretty(bundle['model_name'])}")
            names = bundle["class_names"]
            cm = np.array(results_json["confusion_matrix"])
            cells = pd.DataFrame([(names[i], names[j], int(cm[i, j]))
                                  for i in range(len(names)) for j in range(len(names))],
                                 columns=["actual", "predicted", "count"])
            base = alt.Chart(cells).encode(x=alt.X("predicted:N", sort=names), y=alt.Y("actual:N", sort=names))
            heat = base.mark_rect(cornerRadius=8).encode(
                color=alt.Color("count:Q", scale=alt.Scale(type="symlog", scheme="teals"), legend=None),
                tooltip=["actual", "predicted", "count"])
            text = base.mark_text(fontSize=22, fontWeight="bold").encode(
                text="count:Q",
                color=alt.condition(alt.datum.count > cm.max() / 3, alt.value("#0f172a"), alt.value("#e2e8f0")))
            c1, c2 = st.columns([1.3, 1])
            c1.altair_chart((heat + text).properties(height=300), width="stretch")
            c2.markdown('<div class="note"><br>Rows are the true classes, columns the predictions.<br><br>'
                        'Off-diagonal cells are mistakes: top-right = false alarms (normal traffic flagged), '
                        'bottom-left = missed attacks.</div>', unsafe_allow_html=True)

    # --- ROC, precision-recall, score distribution ------------------------------
    with sub_curves:
        if not curves:
            st.info("No curves found. Run `python -m ml.evaluation --curves` (binary task only).")
        else:
            names_available = list(curves)
            chosen = st.multiselect("Models", names_available, default=names_available, format_func=pretty)
            zoom = st.toggle("Zoom into the top-left corner (the curves are very close to perfect)", value=True)
            roc = pd.concat([pd.DataFrame({"model": pretty(m), "fpr": curves[m]["roc"]["fpr"],
                                           "tpr": curves[m]["roc"]["tpr"]}) for m in chosen]) if chosen else None
            pr = pd.concat([pd.DataFrame({"model": pretty(m), "recall": curves[m]["pr"]["recall"],
                                          "precision": curves[m]["pr"]["precision"]}) for m in chosen]) if chosen else None
            palette = alt.Scale(range=["#22d3ee", "#a78bfa", "#f59e0b", "#22c55e", "#f472b6"])
            if chosen:
                c1, c2 = st.columns(2)
                with c1:
                    ui.section("ROC curve")
                    x_dom, y_dom = ([0, 0.05], [0.9, 1.0]) if zoom else ([0, 1], [0, 1])
                    roc_chart = alt.Chart(roc).mark_line(strokeWidth=2.5, clip=True).encode(
                        x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=x_dom)),
                        y=alt.Y("tpr:Q", title="true positive rate (recall)", scale=alt.Scale(domain=y_dom)),
                        color=alt.Color("model:N", scale=palette, legend=alt.Legend(orient="bottom", columns=3)),
                        tooltip=["model", alt.Tooltip("fpr:Q", format=".4f"), alt.Tooltip("tpr:Q", format=".4f")])
                    diagonal = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(  # random guess
                        strokeDash=[4, 4], color="#475569", clip=True).encode(
                        x=alt.X("fpr:Q", scale=alt.Scale(domain=x_dom)), y=alt.Y("tpr:Q", scale=alt.Scale(domain=y_dom)))
                    st.altair_chart((diagonal + roc_chart).properties(height=340).interactive(), width="stretch")
                with c2:
                    ui.section("Precision-recall curve")
                    dom = [0.9, 1.0] if zoom else [0, 1]
                    pr_chart = alt.Chart(pr).mark_line(strokeWidth=2.5, clip=True).encode(
                        x=alt.X("recall:Q", scale=alt.Scale(domain=dom)),
                        y=alt.Y("precision:Q", scale=alt.Scale(domain=dom)),
                        color=alt.Color("model:N", scale=palette, legend=alt.Legend(orient="bottom", columns=3)),
                        tooltip=["model", alt.Tooltip("recall:Q", format=".4f"), alt.Tooltip("precision:Q", format=".4f")])
                    st.altair_chart(pr_chart.properties(height=340).interactive(), width="stretch")

                auc_table = pd.DataFrame([{"model": pretty(m), "ROC AUC": curves[m]["roc_auc"],
                                           "average precision": curves[m]["average_precision"]} for m in chosen])
                st.dataframe(auc_table.sort_values("ROC AUC", ascending=False), hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.6f")
                                            for c in ["ROC AUC", "average precision"]})
                st.caption("Scroll to zoom and drag to pan the charts. "
                           "The SVM curve uses its decision score because it was trained without probability estimates.")

            with_hist = [m for m in curves if "histogram" in curves[m]]
            if with_hist:
                ui.section("Score distribution: how well does the model separate the classes?")
                c1, c2 = st.columns([1, 3])
                model_h = c1.selectbox("Model", with_hist, format_func=pretty,
                                       index=with_hist.index(bundle["model_name"]) if bundle["model_name"] in with_hist else 0)
                log_scale = c1.toggle("Log scale", value=True)
                c1.markdown('<div class="note">Attack probability given to every test connection, split by its '
                            'true class. Good separation = normal traffic piled up on the left, attacks on the right.</div>',
                            unsafe_allow_html=True)
                h = curves[model_h]["histogram"]
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
            ui.section("Random Forest feature importance (Phase 1 feature selection)")
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
                       f"Random Forest's total importance.")

    # --- training curves -----------------------------------------------------------
    with sub_training:
        nn_results = load_json("results/neural_network.json")
        if nn_results and "history" in nn_results:
            ui.section("Neural network training (PyTorch)")
            hist = nn_results["history"]
            epochs = np.arange(1, len(hist["train_loss"]) + 1)
            loss = pd.concat([pd.DataFrame({"epoch": epochs, "loss": hist["train_loss"], "set": "training"}),
                              pd.DataFrame({"epoch": epochs, "loss": hist["val_loss"], "set": "validation"})])
            best = int(np.argmin(hist["val_loss"])) + 1
            c1, c2 = st.columns(2)
            loss_chart = alt.Chart(loss).mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=2).encode(
                x="epoch:Q", y=alt.Y("loss:Q", title="cross-entropy loss"),
                color=alt.Color("set:N", scale=alt.Scale(range=["#22d3ee", "#f59e0b"]), legend=alt.Legend(orient="bottom", columns=3)),
                tooltip=["epoch", "set", alt.Tooltip("loss:Q", format=".4f")])
            best_rule = alt.Chart(pd.DataFrame({"epoch": [best]})).mark_rule(color=ui.GREEN, strokeDash=[4, 3]).encode(
                x="epoch:Q")
            c1.altair_chart((loss_chart + best_rule).properties(height=300, title="Loss per epoch"), width="stretch")
            acc = pd.DataFrame({"epoch": epochs, "accuracy": hist["val_accuracy"]})
            acc_chart = alt.Chart(acc).mark_line(point=alt.OverlayMarkDef(size=25), color=ui.GREEN, strokeWidth=2).encode(
                x="epoch:Q", y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".1%")),
                tooltip=["epoch", alt.Tooltip("accuracy:Q", format=".2%")])
            c2.altair_chart(acc_chart.properties(height=300, title="Validation accuracy"), width="stretch")
            st.caption(f"Early stopping kept the weights from epoch {best} (green line), where the validation "
                       f"loss was lowest; training stopped {len(epochs) - best} epochs later.")
        if comparison is not None and "train_seconds" in comparison.columns:
            ui.section("Training time per model")
            times = alt.Chart(comparison).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color="#a78bfa").encode(
                x=alt.X("train_seconds:Q", title="seconds"), y=alt.Y("model:N", sort="-x", title=None),
                tooltip=["model", "train_seconds"]).properties(height=200)
            st.altair_chart(times, width="stretch")
            st.caption("Measured on the development laptop; the SVM is trained on a 20,000-row subsample.")

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
