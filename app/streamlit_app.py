"""
streamlit_app.py - Mergence ECCM Platform

Tab 1  Compatibility Simulator   upload → scores → XAI → merge
Tab 2  Pair Analysis             SHAP divergence / blend curve / weights / distributions
Tab 3  About
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
import shap
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
 
# ── Colour helpers ─────────────────────────────────────────────────────────────
def hex_to_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
 
COLOURS = {"PSC": "#4c72b0", "FSC": "#55a868", "RSC": "#c44e52", "ECCM": "#8172b2"}
 
# ── BlendedModel ──────────────────────────────────────────────────────────────
class BlendedModel(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible blended RF - has feature_importances_, predict_proba, predict."""
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
        "fraud":   "./models/epc_model_fraud.pkl",
        "churn":   "./models/epc_model_churn.pkl",
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
    "full":        ("🟢 Full ECCM",    "Real CSV - all metrics fully accurate."),
    "embedded":    ("🟡 Full ECCM",    "Embedded training sample used - accurate."),
    "synthetic":   ("🟠 Partial ECCM", "Synthetic data - FSC is approximate. Upload a CSV for best results."),
    "pscrsc_only": ("🔴 Minimal ECCM", "No data - FSC imputed from history. Only PSC and RSC directly measured."),
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
 
 
# ── SHAP computation ──────────────────────────────────────────────────────────
def compute_shap_values(
    model,
    X: np.ndarray,
    feat_names: list,
    max_explain: int = 200,
) -> tuple:
    """
    Compute SHAP values for a RandomForestClassifier using TreeExplainer.
 
    Performance decisions
    ---------------------
    feature_perturbation="tree_path_dependent" (default for TreeExplainer
    with no background data) is 10-50x faster than the "interventional"
    mode because it uses the training distribution encoded in the tree
    structure rather than marginalising over a background dataset.
    For high-dimensional data (e.g. 2000+ one-hot features) this is the
    only practical choice in an interactive app.
 
    We also cap the number of rows explained at max_explain.
    and mean-|SHAP| charts, 200 samples is sufficient - the visual does
    not meaningfully change with 1400 rows but the compute time does.
 
    Returns:
        shap_vals   np.ndarray shape (n_explain, n_features) - SHAP for class 1
        feat_names  list of feature name strings
    """
    X = np.asarray(X, dtype=np.float64)
 
    # Subsample rows for speed - 200 rows is enough for visual SHAP summaries
    rng = np.random.default_rng(42)
    n   = min(max_explain, len(X))
    X   = X[rng.choice(len(X), size=n, replace=False)]
 
    # No background data = tree_path_dependent mode(fast)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
 
    # sklearn binary RF returns [class0_vals, class1_vals]; class 1 is needed
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]

    return sv.astype(np.float32), feat_names
 
 
 
def shap_divergence_fig(
    sv_a: np.ndarray,
    sv_b: np.ndarray,
    feat_names: list,
    a_n: str,
    b_n: str,
):
    """
    SHAP divergence bar chart.
 
    For each feature: mean|SHAP_A| − mean|SHAP_B|.
    Positive bar = Model A relies on this feature more.
    Negative bar = Model B relies on this feature more.
    Near-zero bar = both models use this feature similarly.
 
    This is a proper data-grounded explanation of why RSC is high or low -
    it shows exactly which features each model prioritises differently.
    """
    mean_abs_a = np.abs(sv_a).mean(axis=0)
    mean_abs_b = np.abs(sv_b).mean(axis=0)
    divergence = mean_abs_a - mean_abs_b
 
    # Sort by absolute divergence, show top 15
    top_idx  = np.argsort(np.abs(divergence))[-15:][::-1]
    div_vals = divergence[top_idx]
    names    = [feat_names[int(i)] if int(i) < len(feat_names) else f"f{int(i)}" for i in top_idx]
 
    colours = [COLOURS["PSC"] if v >= 0 else COLOURS["RSC"] for v in div_vals]
 
    fig = go.Figure(go.Bar(
        x=div_vals,
        y=names,
        orientation="h",
        marker_color=colours,
        text=[f"{v:+.4f}" for v in div_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"SHAP Feature Divergence - {a_n} vs {b_n}",
        xaxis_title=f"← {b_n} relies more     |     {a_n} relies more →",
        height=420,
        margin=dict(t=50, b=30, l=130),
        shapes=[dict(
            type="line", x0=0, x1=0, y0=-0.5, y1=len(top_idx) - 0.5,
            line=dict(color="grey", dash="dash", width=1),
        )],
    )
    return fig
 
 
def shap_mean_abs_comparison_fig(
    sv_a: np.ndarray,
    sv_b: np.ndarray,
    feat_names: list,
    a_n: str,
    b_n: str,
):
    """
    Side-by-side grouped bar chart of mean |SHAP| per feature for both models.
 
    This replaces the old MDI feature importance bar charts with SHAP-based
    importance - SHAP is less biased toward high-cardinality features than MDI.
    """
    mean_abs_a = np.abs(sv_a).mean(axis=0)
    mean_abs_b = np.abs(sv_b).mean(axis=0)
 
    # Union top-15 by combined importance
    combined = mean_abs_a + mean_abs_b
    top_idx  = np.argsort(combined)[-15:][::-1]
    names    = [feat_names[int(i)] if int(i) < len(feat_names) else f"f{int(i)}" for i in top_idx]
 
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=a_n, x=names, y=mean_abs_a[top_idx],
        marker_color=COLOURS["PSC"], opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name=b_n, x=names, y=mean_abs_b[top_idx],
        marker_color=COLOURS["FSC"], opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title=f"Mean |SHAP| - {a_n} vs {b_n}",
        xaxis_tickangle=-35,
        yaxis_title="Mean |SHAP value|",
        height=380,
        margin=dict(t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
 
 
# ── Standard chart builders ────────────────────────────────────────────────────
def scores_bar(s: dict, a_n: str, b_n: str):
    keys = ["PSC", "FSC", "RSC", "ECCM"]
    vals = [s["psc"], s["fsc"], s["rsc"], s["eccm"]]
    fig  = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=[COLOURS[k] for k in keys],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"ECCM Scores - {a_n} + {b_n}",
        yaxis=dict(range=[0, 1.15]),
        height=300, margin=dict(t=45, b=20),
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
    fig  = go.Figure(go.Bar(
        x=keys, y=vals,
        marker_color=[COLOURS[k] for k in keys],
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"ECCM Sub-metric Weights - {task}",
        yaxis=dict(range=[0, 0.75]),
        height=270, margin=dict(t=45, b=20),
    )
    return fig
 
 
def dist_fig(pa, pb, a_n, b_n):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pa, name=a_n, opacity=0.6,
                               marker_color=COLOURS["PSC"], nbinsx=40))
    fig.add_trace(go.Histogram(x=pb, name=b_n, opacity=0.6,
                               marker_color=COLOURS["FSC"], nbinsx=40))
    fig.update_layout(barmode="overlay", height=300,
                      xaxis_title="P(class = 1)", yaxis_title="Count",
                      title="Prediction Score Distributions",
                      margin=dict(t=45, b=20))
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
 
 
# ── XAI narrative ──────────────────────────────────────────────────────────────
def xai_narrative(psc, fsc, rsc, eccm, a_n, b_n, task):
    tier, _, emoji = get_tier(eccm, task)
    p = get_success_probability(eccm, task)
 
    def level(v):
        return "high" if v >= 0.9 else "moderate" if v >= 0.65 else "low"
 
    psc_desc = {
        "high":     "very similar internal structure - both models weight features almost identically.",
        "moderate": "moderately similar structure - some divergence in how each model learned from data.",
        "low":      "quite different structure - the models have learned very different internal representations.",
    }[level(psc)]
    fsc_desc = {
        "high":     "predictions agree very closely on the same inputs.",
        "moderate": "predictions broadly agree, with divergence on harder borderline cases.",
        "low":      "predictions frequently disagree - the models make different calls on the same data.",
    }[level(fsc)]
    rsc_desc = {
        "high":     "nearly identical feature ranking - both models rely on the same features in the same order.",
        "moderate": "broadly similar feature ranking, with some differences in emphasis.",
        "low":      "very different feature priorities - each model relies on a largely different set of signals.",
    }[level(rsc)]
 
    verdict = {
        "High Compatibility":   f"Strong merge candidate - empirical success rate at this ECCM level is **{p:.0%}**.",
        "Medium Compatibility": f"Borderline - empirical success rate is **{p:.0%}**. Check the blend curve before merging.",
        "Low Compatibility":    f"Poor compatibility - empirical success rate is only **{p:.0%}**. Merging is likely to hurt performance.",
    }[tier]
 
    return "\n".join([
        f"**{a_n} + {b_n}** - ECCM **{eccm:.3f}** · {emoji} {tier} · estimated success **{p:.0%}**",
        "",
        f"**PSC {psc:.3f}** - {psc_desc}",
        f"**FSC {fsc:.3f}** - {fsc_desc}",
        f"**RSC {rsc:.3f}** - {rsc_desc}",
        "",
        f"**Verdict:** {verdict}",
    ])
 
 
# ── EPC evidence table ─────────────────────────────────────────────────────────
def epc_table(neighbours):
    if not neighbours:
        st.caption("EPC evidence unavailable - history not loaded.")
        return
    rows = [{
        "#":          n["rank"],
        "Model A":    n.get("model_a", "-"),
        "Model B":    n.get("model_b", "-"),
        "PSC":        f"{n['psc']:.3f}",
        "FSC":        f"{n['fsc']:.3f}",
        "RSC":        f"{n['rsc']:.3f}",
        "Improvement": f"{n['improvement']:+.5f}",
        "Distance":   f"{n['distance']:.4f}",
        "Weight":     f"{n['weight']:.1%}",
    } for n in neighbours]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧬 Mergence")
st.caption("Evolutionary Compatibility & Co-evolution Metric - Model Merge Platform")
 
tab1, tab2, tab3 = st.tabs([
    "🔬 Simulator",
    "📊 Pair Analysis",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("About Mergence")
    st.markdown("""
**Mergence** is a BEng Software Engineering thesis project (IIT / University of Westminster, 2026)
by Trevin Joseph. It proposes **ECCM** - the *Evolutionary Compatibility & Co-evolution Metric* -
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
 
### XAI Components
 
| Component | What it shows |
|-----------|--------------|
| **Mean SHAP Comparison** | Which features matter most to each model (SHAP-based, less biased than MDI) |
| **SHAP Divergence** | Exactly which features each model prioritises differently - grounds the RSC score |
| **EPC Evidence Table** | Which historical merges the EPC prediction is based on |
| **XAI Narrative** | Plain-English verdict on PSC, FSC, RSC, and ECCM |
 
---
 
### Compatibility Tiers (data-driven thresholds)
 
| Tier | Fraud ECCM | Churn ECCM | Empirical P(success) |
|------|-----------|-----------|---------------------|
| ✅ High | ≥ 0.935 | ≥ 0.988 | ≥ 80 % |
| ⚠️ Medium | 0.843 – 0.935 | 0.960 – 0.988 | 40 – 80 % |
| ❌ Low | < 0.843 | < 0.960 | < 40 % |
 
Thresholds derived from isotonic regression on 276 pairs × 5 blend ratios - not guessed.
 
---
 
### Research Questions
 
| | Question | Where addressed |
|-|----------|----------------|
| RQ1 | How to quantify evolutionary pressure? | EPC evidence table - Simulator tab |
| RQ2 | Which PSC/FSC/RSC combination is optimal? | ECCM Weights chart - Pair Analysis tab |
| RQ3 | Efficiency gains over a random baseline? | Tier gate + M2N2 optimisation results |
| RQ4 | Is ECCM interpretable for non-experts? | SHAP divergence + XAI narrative |
 
---
*Trevin Joseph · w1953285 · BEng Software Engineering · IIT / University of Westminster · 2026*
""")
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Compatibility Simulator")
 
    with st.expander("📝 User Manual"):
        st.markdown("""
            1. **Choose your task type** (Fraud, Churn or Unknown). This controls which ECCM weights and compatibility thresholds are applied.
            2. **Upload Model A and Model B** as `.pkl` files. Both should be trained `RandomForestClassifier` objects.
            3. **Optionally upload a validation CSV.** The data quality badge will update to show which mode is active. A labelled CSV unlocks the merge step and the blend curve.
            4. **Click Run Compatibility Check** to compute the ECCM score, tier, EPC evidence and a plain-English explanation.
            5. **Open the Pair Analysis tab** for SHAP explanations, the blend ratio curve and prediction distributions.
            6. **Optionally merge the models** by picking a blend ratio and clicking Merge and Evaluate. You can download the result as a `.pkl`.
        """)
 
    st.divider()
 
    # Task selector
    task = st.selectbox(
        "Task type",
        options=["fraud", "churn", "unknown"],
        format_func=lambda t: {
            "fraud":   "🔍 Fraud Detection",
            "churn":   "📉 Customer Churn",
            "unknown": "❓ Unknown / External",
        }[t],
        help="Controls ECCM weights and compatibility tier thresholds.",
    )
    epc_trainer = load_epc(task)
 
    # Uploads
    st.subheader("Step 1 - Upload")
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
        help="Column in your CSV containing the class labels (0 or 1).",
    )
 
    if not (file_a and file_b):
        st.info("⬆️ Upload Model A and Model B to begin. Validation CSV is optional.")
        st.stop()
 
    ma = joblib.load(io.BytesIO(file_a.read()))
    mb = joblib.load(io.BytesIO(file_b.read()))
 
    up_X, up_y, feat_names = None, None, None
    if val_file:
        vdf = pd.read_csv(val_file)
        if label_col and label_col in vdf.columns:
            feat_names = [c for c in vdf.columns if c != label_col]
            up_X = vdf.drop(columns=[label_col]).astype(np.float64).values
            up_y = vdf[label_col].values
        elif label_col:
            st.warning(f"Column '{label_col}' not found. Available: {vdf.columns.tolist()}")
 
    X_res, data_mode = resolve_data(ma, up_X)
    badge, desc = DATA_MODE_LABELS[data_mode]
    st.info(f"{badge} - {desc}")
 
    # Compatibility check
    st.subheader("Step 2 - Check Compatibility")
 
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
                    "feat": feat_names or [],
                    "task": task,
                    "done": True,
                    # Clear any cached SHAP values from a previous run
                    "shap_a": None, "shap_b": None,
                })
            except Exception as e:
                st.error(f"Computation failed: {e}")
                st.session_state["done"] = False
 
    if not st.session_state.get("done"):
        st.stop()
 
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
        f"{emoji} {tier} - ECCM {ec:.4f}"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;Estimated success probability: {s['p_success']:.0%}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Success probability is empirically calibrated from 276 historical merge experiments, not a guess.")
    st.markdown("")
 
    # Metric cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("PSC  (Structure)",  f"{s['psc']:.4f}")
    mc2.metric("FSC  (Behaviour)",  f"{s['fsc']:.4f}")
    mc3.metric("RSC  (Features)",   f"{s['rsc']:.4f}")
    mc4.metric("ECCM  (Overall)",   f"{ec:.4f}")
    st.caption("All scores 0–1. Higher = more similar. PSC = internal structure, FSC = prediction behaviour, RSC = feature ranking.")
 
    st.plotly_chart(scores_bar(s, a_n, b_n), use_container_width=True)
    st.caption("A bar shorter than the others reveals the weakest compatibility dimension for this pair.")
 
    # EPC evidence
    st.subheader("🔍 EPC Evidence")
    rel = s.get("epc_reliability", 0.5)
    rel_icon = "🟢" if rel >= 0.7 else "🟡" if rel >= 0.4 else "🔴"
    st.caption(
        f"{rel_icon} EPC reliability: {rel:.0%} - "
        + ("close match to historical data." if rel >= 0.7 else
           "moderate match to historical data." if rel >= 0.4 else
           "this pair is unlike past merges - EPC estimate is speculative.")
    )
    epc_table(s.get("epc_neighbours", []))
    st.caption("Each row is a historical merge nearest to this pair in PSC/FSC/RSC space. "
               "EPC is a weighted average of their improvement values.")
 
    # XAI narrative
    st.subheader("🧠 Explanation")
    st.markdown(xai_narrative(s["psc"], s["fsc"], s["rsc"], ec, a_n, b_n, task))
 
    # Merge
    st.divider()
    st.subheader("Step 3 - Merge  (requires labelled CSV)")
 
    if st.session_state["y"] is None:
        st.info("Upload a labelled validation CSV to enable this step.")
    else:
        if tier == "Low Compatibility":
            st.error("⛔ High risk - merging is likely to reduce performance.")
        elif tier == "Medium Compatibility":
            st.warning("⚠️ Moderate risk - check the blend curve on the Pair Analysis tab first.")
 
        proceed = True
        if tier in ("Low Compatibility", "Medium Compatibility"):
            proceed = st.checkbox("I understand the risk and want to merge anyway", key="ack")
 
        if proceed:
            blend_r = st.slider(
                f"Blend weight for {a_n}  (1 − weight → {b_n})",
                0.0, 1.0, 0.5, 0.05,
                help="Use the optimal ratio from the Pair Analysis blend curve.",
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
                rc1.metric(f"{a_n} AUC", f"{auc_a:.6f}")
                rc2.metric(f"{b_n} AUC", f"{auc_b:.6f}")
                rc3.metric("Merged AUC", f"{auc_m:.6f}", delta=f"{delta:+.6f} vs best parent")
                st.caption("Positive delta = the merged model beat the better parent - merge succeeded.")
 
                buf = io.BytesIO()
                joblib.dump(
                    BlendedModel(st.session_state["ma"], st.session_state["mb"], blend_r),
                    buf,
                )
                buf.seek(0)
                st.download_button(
                    "⬇️ Download Merged Model (.pkl)", buf,
                    file_name=f"merged_{a_n}_{b_n}_r{blend_r:.2f}.pkl",
                    mime="application/octet-stream",
                )
                st.caption("The merged model has `predict_proba`, `feature_importances_`, and can be "
                           "re-uploaded into this Simulator.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - PAIR ANALYSIS
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
    feat_ = st.session_state["feat"]
    tier, colour, emoji = s["tier"], s["tier_colour"], s["tier_emoji"]
 
    st.caption(
        f"Showing: **{a_n}** + **{b_n}** · {emoji} {tier} · "
        f"ECCM {ec:.4f} · P(success) ≈ {s['p_success']:.0%}"
    )
 
    # ── Section 1: SHAP Analysis ───────────────────────────────────────────────
    st.subheader("🔬 SHAP Explanations")
 
    if X_ is None:
        st.info("Upload a validation CSV on the Simulator tab to compute SHAP values.")
    else:
        # Compute SHAP values once and cache in session state
        if st.session_state.get("shap_a") is None:
            # Animated progress bar with step labels so the user sees
            # visible progress rather than a frozen screen.
            _prog   = st.progress(0)
            _status = st.empty()
 
            def _show(icon, msg, pct):
                _status.markdown(
                    f"<div style='text-align:center;font-size:1rem;"
                    f"color:#555;padding:10px 0;'>"
                    f"{icon}&nbsp;&nbsp;{msg}</div>",
                    unsafe_allow_html=True,
                )
                _prog.progress(pct)
 
            try:
                _show("🧬", "Analysing model structure…", 10)
                sv_a, fn = compute_shap_values(ma_, X_, feat_)
                _show("📊", "Computing feature contributions for Model B…", 60)
                sv_b, _  = compute_shap_values(mb_, X_, feat_)
                _show("✅", "Building charts…", 95)
                st.session_state["shap_a"] = sv_a
                st.session_state["shap_b"] = sv_b
                st.session_state["shap_feat"] = fn
                _prog.progress(100)
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                st.session_state["shap_a"] = None
            finally:
                _prog.empty()
                _status.empty()
 
        sv_a  = st.session_state.get("shap_a")
        sv_b  = st.session_state.get("shap_b")
        sfeat = st.session_state.get("shap_feat", feat_)
 
        if sv_a is not None and sv_b is not None:
 
            # ── Row 1: Mean |SHAP| comparison ─────────────────────────────
            st.markdown("#### Mean |SHAP| - Feature Importance per Model")
            st.plotly_chart(
                shap_mean_abs_comparison_fig(sv_a, sv_b, sfeat, a_n, b_n),
                use_container_width=True,
            )
            st.caption(
                "Bar height represents the mean absolute SHAP value across all validation samples, "
                "which measures how much each feature shifts predictions on average regardless of direction. "
                "SHAP-based importance is less biased toward high-cardinality features than the built-in "
                "Random Forest MDI importance. "
                "Where both models show a tall bar for the same feature, they are relying on the same signal "
                "and this will be reflected in a higher RSC score. "
                "A tall bar for one model only means that model picked up something the other did not."
            )
 
            st.divider()
 
            # ── Row 3: SHAP divergence ────────────────────────────────────────
            st.markdown("#### SHAP Feature Divergence - Why RSC Is High or Low")
            st.plotly_chart(
                shap_divergence_fig(sv_a, sv_b, sfeat, a_n, b_n),
                use_container_width=True,
            )
 
            # Compute RSC enrichment from SHAP
            shap_rsc = float(np.corrcoef(
                np.abs(sv_a).mean(axis=0),
                np.abs(sv_b).mean(axis=0),
            )[0, 1])
 
            st.caption(
                f"Each bar shows mean|SHAP_A| minus mean|SHAP_B| for that feature. "
                f"Blue bars indicate features that {a_n} relies on more heavily. "
                f"Red bars indicate features that {b_n} prioritises instead. "
                f"Features near zero are treated similarly by both models. "
                f"The SHAP-based RSC correlation for this pair is **{shap_rsc:.3f}** "
                f"(the RSC reported above was {s['rsc']:.3f}, computed from feature importance rankings rather than SHAP values)."
            )
 
    # ── Section 2: Blend Ratio AUC Curve ──────────────────────────────────────
    st.divider()
    st.subheader("Blend Ratio AUC Curve")
    if X_ is not None and y_ is not None:
        try:
            fig_b, br, ba = blend_curve_fig(ma_, mb_, X_, y_, a_n, b_n)
            st.plotly_chart(fig_b, use_container_width=True)
            st.caption(
                f"Each point on the curve shows the merged AUC at a given blend weight. "
                f"The optimal ratio is {br:.2f}, yielding an AUC of {ba:.6f}. "
                f"A flat curve means the exact ratio has little practical effect. "
                f"A sharp peak means one ratio is clearly best and is worth targeting precisely."
            )
        except Exception as e:
            st.warning(f"Could not compute blend curve: {e}")
    else:
        st.info("Upload a labelled CSV on the Simulator tab to see this chart.")
 
    # ── Section 3: ECCM Weights ────────────────────────────────────────────────
    st.divider()
    w = s.get("weights", {})
    if w:
        st.subheader("ECCM Sub-metric Weights")
        st.plotly_chart(weights_bar(w, t), use_container_width=True)
        task_note = {
            "fraud":   "For fraud detection, FSC carries the most weight because prediction agreement turned out to be the strongest predictor of whether a merge would succeed.",
            "churn":   "For churn prediction, the weights are more balanced, with structural signals like PSC and RSC carrying more relative importance.",
            "unknown": "These are the combined weights learned across both fraud and churn experiments.",
        }.get(t, "")        
        st.caption(
            f"A taller bar indicates a stronger predictor of merge success for the {t} task. "
            f"Weights were learned from 1,380 historical merge experiments. {task_note}"
        )

    # ── Section 4: Prediction Distributions ───────────────────────────────────
    if X_ is not None and y_ is not None:
        st.divider()
        st.subheader("Prediction Distributions")
        try:
            pa = ma_.predict_proba(X_)[:, 1]
            pb = mb_.predict_proba(X_)[:, 1]

            st.plotly_chart(dist_fig(pa, pb, a_n, b_n), use_container_width=True)
            st.caption(
                "These histograms show the spread of predicted class probabilities across the validation set. "
                "When both distributions look similar in shape, the models are behaving alike, which corresponds to a high FSC score. "
                "Models with peaks near 0 and 1 are making more decisive predictions than those clustering in the middle range."
            )       

            st.plotly_chart(scatter_fig(pa, pb, y_, a_n, b_n), use_container_width=True)
            st.caption(
                "Each point represents one validation sample, plotted by the probability each model assigned to it. "
                "Points on the diagonal mean both models agreed. "
                "Points off the diagonal are cases where the models disagreed, and these are the samples most sensitive to the blend ratio you choose. "
                "Red points are class 1 instances (fraud or churn). Blue points are class 0."
            )
        except Exception as e:
            st.warning(f"Distribution plots failed: {e}")
