"""
streamlit_app.py — Mergence ECCM Platform

Tab 1  Compatibility Simulator   upload → scores → XAI → merge
Tab 2  Pair Analysis             radar / blend curve / features / distributions
Tab 3  About & How To Use

UX decisions in this version:
  - Feature importance charts removed from Tab 1 (they live in Tab 2 — no duplication)
  - Tab 1 is kept to: upload → badge → scores → EPC table → XAI → merge only
  - Tab 2 groups all visual analysis; every chart has a short caption
  - Captions are one sentence maximum — just enough to orient a first-time reader
  - Onboarding is a single collapsed expander so it doesn't clutter the page
  - No repeated dividers between every section — used only where sections really change
  - hex_to_rgba() fixes the Plotly scatterpolar fillcolor crash
"""

import io
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from metrics.eccm import (
    ECCMCalculator,
    get_tier,
    get_success_probability,
    synthetic_validation_from_rf,
)
from metrics.epc import EPCTrainer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mergence – ECCM Platform",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
/* Add vertical breathing room between all Streamlit widget blocks */
div[data-testid="stVerticalBlock"] > div {
    margin-bottom: 0.75rem;
}
/* Extra space above subheaders */
h3 {
    margin-top: 1.25rem !important;
}
/* Loosen up file uploader and text input spacing */
div[data-testid="stFileUploader"],
div[data-testid="stTextInput"] {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Colour helper ──────────────────────────────────────────────────────────────
def hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """'#rrggbb' → 'rgba(r,g,b,alpha)'  — Plotly scatterpolar requires rgba, not 8-digit hex."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Inline colour map ──────────────────────────────────────────────────────────
COLOURS = {"PSC": "#4c72b0", "FSC": "#55a868", "RSC": "#c44e52", "ECCM": "#8172b2"}

# ── BlendedModel ──────────────────────────────────────────────────────────────
class BlendedModel(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible blended RF pair — has feature_importances_, predict_proba, predict."""
    def __init__(self, model_a=None, model_b=None, ratio: float = 0.5):
        self.model_a, self.model_b, self.ratio = model_a, model_b, ratio

    def predict_proba(self, X):
        b = self.ratio * self.model_a.predict_proba(X)[:, 1] + \
            (1 - self.ratio) * self.model_b.predict_proba(X)[:, 1]
        return np.column_stack([1 - b, b])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.ratio * self.model_a.feature_importances_ + \
               (1 - self.ratio) * self.model_b.feature_importances_

    @property
    def classes_(self):        return self.model_a.classes_
    @property
    def n_features_in_(self):  return self.model_a.n_features_in_
    @property
    def X_train_sample_(self): return getattr(self.model_a, "X_train_sample_", None)

# ── EPC loader ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_epc(task: str) -> EPCTrainer:
    paths = {
        "fraud": "./models/epc_model_fraud.pkl",
        "churn": "./models/epc_model_churn.pkl",
        "unknown": "./models/epc_model.pkl",
    }
    epc = EPCTrainer(k=5)
    for p in [paths.get(task, "./models/epc_model.pkl"), "./models/epc_model.pkl"]:
        if Path(p).exists():
            try:
                epc.load(p)
                return epc
            except Exception:
                continue
    return epc

# ── Data resolution ───────────────────────────────────────────────────────────
DATA_MODE_LABELS = {
    "full":        ("🟢 Full ECCM",     "Real CSV — all metrics fully accurate."),
    "embedded":    ("🟡 Full ECCM",     "Embedded training sample used — accurate."),
    "synthetic":   ("🟠 Partial ECCM",  "Synthetic data — FSC is approximate. Upload a CSV for best results."),
    "pscrsc_only": ("🔴 Minimal ECCM",  "No data — FSC imputed from history. Only PSC and RSC directly measured."),
}

def resolve_data(model_a, uploaded_X):
    if uploaded_X is not None and len(uploaded_X) > 0:
        return uploaded_X, "full"
    sample = getattr(model_a, "X_train_sample_", None)
    if sample is not None:
        return sample, "embedded"
    try:
        return synthetic_validation_from_rf(model_a), "synthetic"
    except Exception:
        return None, "pscrsc_only"

# ── Chart builders ────────────────────────────────────────────────────────────
def scores_bar(s: dict, a_n: str, b_n: str):
    keys = ["PSC", "FSC", "RSC", "ECCM"]
    vals = [s["psc"], s["fsc"], s["rsc"], s["eccm"]]
    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=[COLOURS[k] for k in keys],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"ECCM Scores — {a_n} + {b_n}",
        yaxis=dict(range=[0, 1.15]),
        height=300, margin=dict(t=45, b=20),
    )
    return fig


def fi_chart(model, name, feat_names):
    if not hasattr(model, "feature_importances_"):
        return None
    fi  = model.feature_importances_
    nms = feat_names if feat_names and len(feat_names) == len(fi) else [f"f{i}" for i in range(len(fi))]
    df  = pd.DataFrame({"feature": nms, "importance": fi}).nlargest(15, "importance")
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title=f"Top-15 Features — {name}",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=360, showlegend=False,
                      margin=dict(t=45, b=20))
    return fig


def radar_chart(s: dict, colour: str, a_n: str, b_n: str):
    cats = ["PSC", "FSC", "RSC", "ECCM"]
    vals = [s["psc"], s["fsc"], s["rsc"], s["eccm"]]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor=hex_to_rgba(colour, 0.18),
        line_color=colour,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, 1],
                tickfont=dict(color="rgba(255,255,255,0.75)", size=10),
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],   # explicit ticks so none are skipped
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(color="rgba(255,255,255,0.9)", size=12),
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.15)",
            ),
            bgcolor="rgba(0,0,0,0)",  # transparent polar background
        ),
        paper_bgcolor="rgba(0,0,0,0)",  # transparent chart background
        title=dict(text=f"{a_n} + {b_n}", font=dict(color="white")),
        height=340, showlegend=False,
        margin=dict(t=45, b=20),
    )
    return fig


def blend_curve_fig(ma, mb, X, y, a_n, b_n):
    pa = ma.predict_proba(X)[:, 1]
    pb = mb.predict_proba(X)[:, 1]
    rs = np.linspace(0, 1, 21)
    au = [roc_auc_score(y, r * pa + (1 - r) * pb) for r in rs]
    br = rs[int(np.argmax(au))]
    fig = px.line(x=rs, y=au,
                  labels={"x": f"← 100% {b_n}   Weight on {a_n}   100% {a_n} →", "y": "AUC"},
                  title=f"Blend Ratio vs AUC  (best r = {br:.2f})")
    fig.add_vline(x=br, line_dash="dash", line_color="green",
                  annotation_text=f"r = {br:.2f}", annotation_position="top right")
    fig.update_layout(height=330, margin=dict(t=45, b=20))
    return fig, br, max(au)


def weights_bar(w: dict, task: str):
    keys = ["PSC", "FSC", "RSC"]
    vals = [w["w_psc"], w["w_fsc"], w["w_rsc"]]
    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=[COLOURS[k] for k in keys],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"ECCM Sub-metric Weights — {task}",
        yaxis=dict(range=[0, 0.75]),
        height=270, margin=dict(t=45, b=20),
    )
    return fig


def dist_fig(pa, pb, a_n, b_n):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pa, name=a_n, opacity=0.6, marker_color=COLOURS["PSC"], nbinsx=40))
    fig.add_trace(go.Histogram(x=pb, name=b_n, opacity=0.6, marker_color=COLOURS["FSC"], nbinsx=40))
    fig.update_layout(barmode="overlay", height=300,
                      xaxis_title="P(class = 1)", yaxis_title="Count",
                      title="Prediction Score Distributions", margin=dict(t=45, b=20))
    return fig


def scatter_fig(pa, pb, y, a_n, b_n):
    fig = px.scatter(x=pa, y=pb, opacity=0.25,
                     labels={"x": f"{a_n} score", "y": f"{b_n} score"},
                     title="Prediction Agreement",
                     color=y.astype(str),
                     color_discrete_map={"0": COLOURS["PSC"], "1": COLOURS["RSC"]})
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="grey"),
                             name="Perfect agreement"))
    fig.update_layout(height=360, margin=dict(t=45, b=20))
    return fig

# ── XAI text ──────────────────────────────────────────────────────────────────
def xai_narrative(psc, fsc, rsc, eccm, a_n, b_n, task):
    tier, _, emoji = get_tier(eccm, task)
    p = get_success_probability(eccm, task)

    def level(v):
        return "high" if v >= 0.9 else "moderate" if v >= 0.65 else "low"

    psc_desc = {
        "high":     "very similar internal structure — both models weight features almost identically.",
        "moderate": "moderately similar structure — some divergence in how each model learned from data.",
        "low":      "quite different structure — the models have learned very different internal representations.",
    }[level(psc)]
    fsc_desc = {
        "high":     "predictions agree very closely on the same inputs.",
        "moderate": "predictions broadly agree, with divergence on harder borderline cases.",
        "low":      "predictions frequently disagree — the models make different calls on the same data.",
    }[level(fsc)]
    rsc_desc = {
        "high":     "nearly identical feature ranking — both models rely on the same features in the same order.",
        "moderate": "broadly similar feature ranking, with some differences in emphasis.",
        "low":      "very different feature priorities — each model relies on a largely different set of signals.",
    }[level(rsc)]

    verdict = {
        "High Compatibility":   f"Strong merge candidate — empirical success rate at this ECCM level is **{p:.0%}**.",
        "Medium Compatibility": f"Borderline — empirical success rate is **{p:.0%}**. Check the blend curve before merging.",
        "Low Compatibility":    f"Poor compatibility — empirical success rate is only **{p:.0%}**. Merging is likely to hurt performance.",
    }[tier]

    return "\n".join([
        f"**{a_n} + {b_n}** — ECCM **{eccm:.3f}** · {emoji} {tier} · estimated success **{p:.0%}**",
        "",
        f"**PSC {psc:.3f}** — {psc_desc}",
        f"**FSC {fsc:.3f}** — {fsc_desc}",
        f"**RSC {rsc:.3f}** — {rsc_desc}",
        "",
        f"**Verdict:** {verdict}",
    ])

# ── EPC evidence table ────────────────────────────────────────────────────────
def epc_table(neighbours):
    if not neighbours:
        st.caption("EPC evidence unavailable — history not loaded.")
        return
    rows = [{
        "#": n["rank"],
        "Model A": n.get("model_a", "—"), "Model B": n.get("model_b", "—"),
        "PSC": f"{n['psc']:.3f}", "FSC": f"{n['fsc']:.3f}", "RSC": f"{n['rsc']:.3f}",
        "Improvement": f"{n['improvement']:+.5f}",
        "Distance": f"{n['distance']:.4f}", "Weight": f"{n['weight']:.1%}",
    } for n in neighbours]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧬 Mergence")
st.caption("Evolutionary Compatibility & Co-evolution Metric — Model Merge Platform")

tab1, tab2, tab3 = st.tabs([
    "🔬 Simulator",
    "📊 Pair Analysis",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Compatibility Simulator")

    with st.expander("📑 User Manual"):
        st.markdown("""
1. **Pick your task** (Fraud / Churn / Unknown) — this sets the scoring weights and tier thresholds.
2. **Upload Model A and Model B** — `.pkl` files from a trained `RandomForestClassifier`.
3. **Optionally upload a validation CSV** — the coloured badge below the uploader tells you which data mode is active.  
   No CSV? The app uses a sample embedded in the model automatically.
4. **Click ▶ Run Compatibility Check** — you get a tier badge, four scores, an EPC evidence table, and a plain-English explanation.
5. **Visit 📊 Pair Analysis** for deeper charts (radar, blend curve, feature charts, prediction distributions).
6. **Optionally merge** — use the slider to pick a blend ratio, click Merge, download the result.
        """)

    st.divider()

    # ── Task selector ─────────────────────────────────────────────────────────
    task = st.selectbox(
        "Task type",
        options=["fraud", "churn", "unknown"],
        format_func=lambda t: {
            "fraud": "🔍 Fraud Detection",
            "churn": "📉 Customer Churn",
            "unknown": "❓ Unknown / External",
        }[t],
        help="Controls ECCM weights and compatibility tier thresholds.",
    )
    epc_trainer = load_epc(task)

    # ── Uploads ───────────────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload")
    c1, c2 = st.columns(2)
    with c1:
        file_a = st.file_uploader("Model A (.pkl)", type=["pkl"], key="fa")
        a_name = st.text_input("Label", value="Model A", key="an")
    with c2:
        file_b = st.file_uploader("Model B (.pkl)", type=["pkl"], key="fb")
        b_name = st.text_input("Label", value="Model B", key="bn")

    val_file  = st.file_uploader("Validation CSV (optional)", type=["csv"], key="fv")
    label_col = st.text_input(
        "Label column name",
        value={"fraud": "Class", "churn": "Churn"}.get(task, ""),
        key=f"lc_{task}",
        help="The column in your CSV that contains the class labels (0 or 1).",
    )

    if not (file_a and file_b):
        st.info("⬆️ Upload Model A and Model B to begin. Validation CSV is optional.")
        st.stop()

    # Load models
    ma = joblib.load(io.BytesIO(file_a.read()))
    mb = joblib.load(io.BytesIO(file_b.read()))

    # Parse optional CSV
    up_X, up_y, feat_names = None, None, None
    if val_file:
        vdf = pd.read_csv(val_file)
        if label_col and label_col in vdf.columns:
            feat_names = [c for c in vdf.columns if c != label_col]
            up_X = vdf.drop(columns=[label_col]).values
            up_y = vdf[label_col].values
        elif label_col:
            st.warning(f"Column '{label_col}' not found. Available: {vdf.columns.tolist()}")

    X_res, data_mode = resolve_data(ma, up_X)
    badge, desc = DATA_MODE_LABELS[data_mode]
    st.info(f"{badge} — {desc}")

    # ── Compatibility check ───────────────────────────────────────────────────
    st.subheader("Step 2 — Check Compatibility")

    if st.button("▶ Run Compatibility Check", type="primary", key="run"):
        with st.spinner("Computing ECCM…"):
            try:
                calc = ECCMCalculator(task=task)
                calc.epc = epc_trainer
                scores = calc.compute(ma, mb, X=X_res)
                st.session_state.update({
                    "scores": scores, "ec": scores["eccm"],
                    "ma": ma, "mb": mb,
                    "X": X_res, "y": up_y,
                    "a_n": a_name, "b_n": b_name,
                    "feat": feat_names or [], "task": task,
                    "done": True,
                })
            except Exception as e:
                st.error(f"Computation failed: {e}")
                st.session_state["done"] = False

    if not st.session_state.get("done"):
        st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    s   = st.session_state["scores"]
    ec  = st.session_state["ec"]
    a_n = st.session_state["a_n"]
    b_n = st.session_state["b_n"]
    tier, colour, emoji = s["tier"], s["tier_colour"], s["tier_emoji"]

    # Tier banner
    st.markdown(
        f"<div style='padding:12px 18px;border-radius:8px;"
        f"background:{hex_to_rgba(colour,0.12)};border-left:5px solid {colour};"
        f"font-size:1.1rem;font-weight:600;'>"
        f"{emoji} {tier} — ECCM {ec:.4f}"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;Estimated success probability: {s['p_success']:.0%}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Success probability is empirically calibrated from 276 historical merge experiments for this task — not a guess.")
    st.markdown("")

    # Four metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("PSC  (Structure)",  f"{s['psc']:.4f}")
    mc2.metric("FSC  (Behaviour)",  f"{s['fsc']:.4f}")
    mc3.metric("RSC  (Features)",   f"{s['rsc']:.4f}")
    mc4.metric("ECCM  (Overall)",   f"{ec:.4f}")
    st.caption("All scores 0–1. Higher = more similar. PSC = internal structure, FSC = prediction behaviour, RSC = feature ranking.")

    # Scores bar
    st.plotly_chart(scores_bar(s, a_n, b_n), use_container_width=True)
    st.caption("A bar that is noticeably shorter than the others reveals the weakest dimension of compatibility for this pair.")

    # ── EPC evidence ──────────────────────────────────────────────────────────
    st.subheader("🔍 EPC Evidence")
    rel = s.get("epc_reliability", 0.5)
    rel_icon = "🟢" if rel >= 0.7 else "🟡" if rel >= 0.4 else "🔴"
    st.caption(f"{rel_icon} EPC reliability: {rel:.0%} — "
               + ("close match to historical data." if rel >= 0.7 else
                  "moderate match to historical data." if rel >= 0.4 else
                  "this pair is unlike past merges — EPC estimate is speculative."))
    epc_table(s.get("epc_neighbours", []))
    st.caption("Each row is a historical merge whose PSC/FSC/RSC scores were nearest to this pair. "
               "The EPC prediction is a weighted average of their Improvement values.")

    # ── XAI explanation ───────────────────────────────────────────────────────
    st.subheader("🧠 Explanation")
    st.markdown(xai_narrative(s["psc"], s["fsc"], s["rsc"], ec, a_n, b_n, task))

    # ── Merge ─────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 3 — Merge  (requires labelled CSV)")

    if st.session_state["y"] is None:
        st.info("Upload a labelled validation CSV to enable this step.")
    else:
        if tier == "Low Compatibility":
            st.error("⛔ High risk — merging is likely to reduce performance.")
        elif tier == "Medium Compatibility":
            st.warning("⚠️ Moderate risk — check the blend curve on the Pair Analysis tab first.")

        proceed = True
        if tier in ("Low Compatibility", "Medium Compatibility"):
            proceed = st.checkbox("I understand the risk and want to merge anyway", key="ack")

        if proceed:
            blend_r = st.slider(
                f"Blend weight for {a_n}  (1 − weight → {b_n})",
                0.0, 1.0, 0.5, 0.05,
                help="0.5 = equal weight. Use the optimal ratio from the Pair Analysis blend curve.",
            )

            if st.button("⚗️ Merge & Evaluate", type="primary", key="merge"):
                with st.spinner("Merging…"):
                    X_m, y_m = st.session_state["X"], st.session_state["y"]
                    pa_ = st.session_state["ma"].predict_proba(X_m)[:, 1]
                    pb_ = st.session_state["mb"].predict_proba(X_m)[:, 1]
                    auc_a = roc_auc_score(y_m, pa_)
                    auc_b = roc_auc_score(y_m, pb_)
                    auc_m = roc_auc_score(y_m, blend_r * pa_ + (1 - blend_r) * pb_)
                    delta = auc_m - max(auc_a, auc_b)

                st.success(f"Merged AUC: **{auc_m:.6f}**")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric(f"{a_n} AUC",  f"{auc_a:.6f}")
                rc2.metric(f"{b_n} AUC",  f"{auc_b:.6f}")
                rc3.metric("Merged AUC",  f"{auc_m:.6f}", delta=f"{delta:+.6f} vs best parent")
                st.caption("A positive delta means the merged model beat the better parent — the merge succeeded.")

                buf = io.BytesIO()
                joblib.dump(BlendedModel(st.session_state["ma"], st.session_state["mb"], blend_r), buf)
                buf.seek(0)
                st.download_button(
                    "⬇️ Download Merged Model (.pkl)", buf,
                    file_name=f"merged_{a_n}_{b_n}_r{blend_r:.2f}.pkl",
                    mime="application/octet-stream",
                )
                st.caption("The merged model is sklearn-compatible — it has `predict_proba`, `feature_importances_`, "
                           "and can be re-uploaded into this Simulator.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAIR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Pair Analysis")

    if not st.session_state.get("done"):
        st.info("Run a compatibility check on the **🔬 Simulator** tab first.")
        st.stop()

    s   = st.session_state["scores"]
    ec  = st.session_state["ec"]
    a_n = st.session_state["a_n"]
    b_n = st.session_state["b_n"]
    t   = st.session_state["task"]
    ma_ = st.session_state["ma"]
    mb_ = st.session_state["mb"]
    X_  = st.session_state["X"]
    y_  = st.session_state["y"]
    tier, colour, emoji = s["tier"], s["tier_colour"], s["tier_emoji"]

    st.caption(f"Showing: **{a_n}** + **{b_n}** · {emoji} {tier} · ECCM {ec:.4f} · P(success) ≈ {s['p_success']:.0%}")

    # ── Row 1: Radar + Blend curve ────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("ECCM Radar")
        st.plotly_chart(radar_chart(s, colour, a_n, b_n), use_container_width=True)
        st.caption(
            "Each axis is one sub-metric (0 = centre, 1 = outer edge). "
            "A large, even polygon = well-rounded compatibility. "
            "A squashed axis pinpoints the weakest dimension."
        )

    with r1c2:
        st.subheader("Blend Ratio AUC Curve")
        if X_ is not None and y_ is not None:
            try:
                fig_b, br, ba = blend_curve_fig(ma_, mb_, X_, y_, a_n, b_n)
                st.plotly_chart(fig_b, use_container_width=True)
                st.caption(
                    f"Each point is the merged AUC at a specific blend ratio. "
                    f"The green line marks the optimal ratio ({br:.2f}, AUC {ba:.6f}). "
                    f"A flat curve means the ratio choice barely matters."
                )
            except Exception as e:
                st.warning(f"Could not compute blend curve: {e}")
        else:
            st.info("Upload a labelled CSV on the Simulator tab to see this chart.")

    # ── Row 2: ECCM Weights ───────────────────────────────────────────────────
    st.divider()
    w = s.get("weights", {})
    if w:
        st.subheader("ECCM Sub-metric Weights")
        st.plotly_chart(weights_bar(w, t), use_container_width=True)
        task_note = {
            "fraud":   "For fraud, FSC dominates — prediction agreement is the strongest predictor of merge success.",
            "churn":   "For churn, weights are more balanced — structural signals (PSC, RSC) carry more weight.",
            "unknown": "Combined weights used (learned from both fraud and churn experiments).",
        }.get(t, "")
        st.caption(
            f"Taller bar = that sub-metric is a better predictor of merge success for **{t}**. "
            f"Weights were learned from 1 380 historical experiments. {task_note}"
        )

    # ── Row 3: Feature importances (ONLY here, not in Tab 1) ──────────────────
    st.divider()
    st.subheader("Feature Importance Comparison")
    st.caption(
        "Longer bar = the model relies more heavily on that feature. "
        "Matching top features across both charts = high RSC score. "
        "Different top features = complementary signals, potentially useful when merging."
    )
    fc1, fc2 = st.columns(2)
    with fc1:
        fig = fi_chart(ma_, a_n, st.session_state["feat"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature_importances_ on Model A.")
    with fc2:
        fig = fi_chart(mb_, b_n, st.session_state["feat"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature_importances_ on Model B.")

    if hasattr(ma_, "feature_importances_") and hasattr(mb_, "feature_importances_"):
        top_a   = set(np.argsort(ma_.feature_importances_)[-10:])
        top_b   = set(np.argsort(mb_.feature_importances_)[-10:])
        overlap = len(top_a & top_b)
        st.caption(
            f"**Top-10 feature overlap: {overlap}/10.** "
            + ("High — same signals, low merge risk." if overlap >= 7 else
               "Moderate — some shared, some unique signals." if overlap >= 4 else
               "Low — very different signals; key driver of low RSC.")
        )

    # ── Row 4: Prediction distributions (only if labels available) ────────────
    if X_ is not None and y_ is not None:
        st.divider()
        st.subheader("Prediction Distributions")
        try:
            pa = ma_.predict_proba(X_)[:, 1]
            pb = mb_.predict_proba(X_)[:, 1]

            st.plotly_chart(dist_fig(pa, pb, a_n, b_n), use_container_width=True)
            st.caption(
                "Overlaid histograms of each model's predicted probability for class 1. "
                "Similar shapes = models behave alike (high FSC). "
                "Models that peak near 0 and 1 are more decisive than those clustering in the middle."
            )

            st.plotly_chart(scatter_fig(pa, pb, y_, a_n, b_n), use_container_width=True)
            st.caption(
                "Each dot = one validation sample. "
                "Dots on the diagonal = both models gave the same probability (perfect agreement). "
                "Off-diagonal dots = disagreement — these samples are most sensitive to the blend ratio. "
                "Red = class 1 (fraud / churn), blue = class 0."
            )
        except Exception as e:
            st.warning(f"Distribution plots failed: {e}")

    # ── XAI narrative ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧠 XAI Narrative")
    st.markdown(xai_narrative(s["psc"], s["fsc"], s["rsc"], ec, a_n, b_n, t))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("About Mergence")
    st.markdown("""
**Mergence** is a BEng Software Engineering thesis project (IIT / University of Westminster, 2026)
by Trevin Joseph. It proposes **ECCM** — the *Evolutionary Compatibility & Co-evolution Metric* —
a composite score that predicts, before execution, whether merging two trained models will improve performance.

---

### ECCM Formula

`ECCM = w_psc × PSC + w_fsc × FSC + w_rsc × RSC + w_epc × EPC`

| Sub-metric | Measures |
|-----------|---------|
| **PSC** | Cosine similarity of feature importance vectors |
| **FSC** | Pearson correlation of prediction probabilities |
| **RSC** | Spearman rank correlation of feature importance rankings |
| **EPC** | k-NN weighted average improvement from similar historical merges |

Weights are **task-specific**, learned from 1 380 historical merge experiments.

---

### Compatibility Tiers (data-driven thresholds)

| Tier | Fraud ECCM | Churn ECCM | Empirical P(success) |
|------|-----------|-----------|---------------------|
| ✅ High | ≥ 0.935 | ≥ 0.988 | ≥ 80 % |
| ⚠️ Medium | 0.843 – 0.935 | 0.960 – 0.988 | 40 – 80 % |
| ❌ Low | < 0.843 | < 0.960 | < 40 % |

Thresholds derived from isotonic regression on 276 pairs × 5 blend ratios — not chosen arbitrarily.

---

### Research Questions

| | Question | Where addressed |
|-|----------|----------------|
| RQ1 | How to quantify evolutionary pressure? | EPC evidence table — Simulator tab |
| RQ2 | Which PSC/FSC/RSC combination is optimal? | ECCM Weights chart — Pair Analysis tab |
| RQ3 | Efficiency gains over a random baseline? | Tier gate + M2N2 optimisation results |
| RQ4 | Is ECCM interpretable for non-experts? | XAI narrative + chart captions throughout |

---
*Trevin Joseph · w1953285 · BEng Software Engineering · IIT / University of Westminster · 2026*
""")