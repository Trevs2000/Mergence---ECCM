"""
streamlit_app.py — Mergence ECCM Platform

Three tabs:
  Tab 1 — Compatibility Simulator  (upload models → ECCM → XAI → merge)
  Tab 2 — Pair Analysis             (radar, blend curve, distributions, weights)
  Tab 3 — About

Key design decisions:
  - CSV upload is optional (fallback chain: embedded sample → synthetic)
  - Task selector loads per-task ECCM weights and tier thresholds
  - EPC evidence table shows k-NN neighbours for interpretability (RQ4)
  - BlendedModel is sklearn-compatible (can be re-uploaded as Model A/B)
  - Tier banner shows calibrated P(success), not just a colour label
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
from metrics.psc import PSCCalculator
from metrics.fsc import FSCCalculator
from metrics.rsc import RSCCalculator

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mergence – ECCM Platform", page_icon="🧬", layout="wide")

# ── BlendedModel (sklearn-compatible) ─────────────────────────────────────────
class BlendedModel(BaseEstimator, ClassifierMixin):
    """
    Lightweight sklearn-compatible wrapper for a blended RF pair.
    Exposes feature_importances_, classes_, n_features_in_ and
    X_train_sample_ so it can be re-uploaded into the Simulator.
    """
    def __init__(self, model_a=None, model_b=None, ratio: float = 0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.ratio   = ratio

    def predict_proba(self, X):
        pa = self.model_a.predict_proba(X)[:, 1]
        pb = self.model_b.predict_proba(X)[:, 1]
        b  = self.ratio * pa + (1 - self.ratio) * pb
        return np.column_stack([1 - b, b])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.ratio * self.model_a.feature_importances_ + \
               (1 - self.ratio) * self.model_b.feature_importances_

    @property
    def classes_(self):          return self.model_a.classes_
    @property
    def n_features_in_(self):    return self.model_a.n_features_in_
    @property
    def X_train_sample_(self):   return getattr(self.model_a, "X_train_sample_", None)


# ── EPC loader (cached per task) ──────────────────────────────────────────────
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
DATA_MODE_INFO = {
    "full":        ("🟢 Full ECCM",    "Real validation CSV — all metrics are accurate."),
    "embedded":    ("🟡 Full ECCM",    "Embedded training sample used — accurate."),
    "synthetic":   ("🟠 Partial ECCM", "Synthetic data from tree thresholds — FSC is approximate."),
    "pscrsc_only": ("🔴 Minimal ECCM", "No data available — FSC imputed from history."),
}

def resolve_data(model_a, uploaded_X) -> tuple:
    if uploaded_X is not None and len(uploaded_X) > 0:
        return uploaded_X, "full"
    sample = getattr(model_a, "X_train_sample_", None)
    if sample is not None:
        return sample, "embedded"
    try:
        return synthetic_validation_from_rf(model_a), "synthetic"
    except Exception:
        return None, "pscrsc_only"


# ── Chart helpers ─────────────────────────────────────────────────────────────
def fi_chart(model, name, feat_names):
    if not hasattr(model, "feature_importances_"):
        return None
    fi  = model.feature_importances_
    n   = len(fi)
    nms = feat_names if feat_names and len(feat_names) == n else [f"f{i}" for i in range(n)]
    df  = pd.DataFrame({"feature": nms, "importance": fi}).nlargest(15, "importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 title=f"Top-15 Features — {name}",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
    return fig


def blend_curve(ma, mb, X, y, a_n, b_n):
    pa = ma.predict_proba(X)[:, 1]
    pb = mb.predict_proba(X)[:, 1]
    rs = np.linspace(0, 1, 21)
    au = [roc_auc_score(y, r * pa + (1 - r) * pb) for r in rs]
    br = rs[int(np.argmax(au))]
    fig = px.line(x=rs, y=au,
                  labels={"x": f"← 100% {b_n}  Weight on {a_n}  100% {a_n} →", "y": "AUC"},
                  title=f"AUC across blend ratios (optimal r={br:.2f})")
    fig.add_vline(x=br, line_dash="dash", line_color="green",
                  annotation_text=f"r={br:.2f}", annotation_position="top right")
    fig.update_layout(height=350)
    return fig, br, max(au)


def xai_narrative(psc, fsc, rsc, eccm, a_n, b_n, task):
    tier, _, emoji = get_tier(eccm, task)
    p = get_success_probability(eccm, task)
    lines = [
        f"**{a_n}** and **{b_n}** — ECCM = **{eccm:.3f}** "
        f"({emoji} {tier}, estimated success: **{p:.0%}**)",
        "",
        f"**PSC = {psc:.3f}:** "
        + ("Very similar internal weights." if psc >= 0.9 else
           "Moderately similar weights." if psc >= 0.65 else "Different weight structures."),
        f"**FSC = {fsc:.3f}:** "
        + ("Predictions agree closely." if fsc >= 0.9 else
           "Predictions agree on most cases." if fsc >= 0.65 else "Predictions frequently disagree."),
        f"**RSC = {rsc:.3f}:** "
        + ("Nearly identical feature ranking." if rsc >= 0.9 else
           "Broadly similar feature ranking." if rsc >= 0.65 else "Very different feature priorities."),
        "",
        f"**Verdict:** " + {
            "High Compatibility":   f"Strong merge candidate. Empirical success ≈ {p:.0%}.",
            "Medium Compatibility": f"Borderline. Check the blend curve. Empirical success ≈ {p:.0%}.",
            "Low Compatibility":    f"Poor compatibility. Merging likely reduces performance (success ≈ {p:.0%}).",
        }[tier],
    ]
    return "\n".join(lines)


def epc_table(neighbours):
    if not neighbours:
        st.caption("No EPC evidence (history not loaded).")
        return
    rows = [{
        "Rank": n["rank"],
        "Model A": n.get("model_a", "—"), "Model B": n.get("model_b", "—"),
        "PSC": f"{n['psc']:.3f}", "FSC": f"{n['fsc']:.3f}", "RSC": f"{n['rsc']:.3f}",
        "Improvement": f"{n['improvement']:+.5f}",
        "Distance": f"{n['distance']:.4f}", "Weight": f"{n['weight']:.1%}",
    } for n in neighbours]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
st.title("🧬 Mergence")
st.caption("Evolutionary Compatibility & Co-evolution Metric — Model Merge Platform")

tab1, tab2, tab3 = st.tabs(["🔬 Compatibility Simulator", "📊 Analysis", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COMPATIBILITY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Model Compatibility Simulator")

    # Task selector
    task = st.selectbox(
        "Task type (sets ECCM weights and tier thresholds)",
        options=["fraud", "churn", "unknown"],
        format_func=lambda t: {"fraud": "🔍 Fraud Detection",
                                "churn": "📉 Customer Churn",
                                "unknown": "❓ Unknown"}[t],
    )
    epc_trainer = load_epc(task)

    # Uploads
    st.subheader("Step 1 — Upload Models")
    c1, c2 = st.columns(2)
    with c1:
        file_a = st.file_uploader("Model A (.pkl)", type=["pkl"], key="fa")
        a_name = st.text_input("Label for Model A", value="Model A", key="an")
    with c2:
        file_b = st.file_uploader("Model B (.pkl)", type=["pkl"], key="fb")
        b_name = st.text_input("Label for Model B", value="Model B", key="bn")

    val_file  = st.file_uploader("Validation CSV (optional)", type=["csv"], key="fv")
    label_col = st.text_input("Label column name", value="Class", key="lc")

    if file_a and file_b:
        ma = joblib.load(io.BytesIO(file_a.read()))
        mb = joblib.load(io.BytesIO(file_b.read()))

        up_X, up_y, feat_names = None, None, None
        if val_file:
            vdf = pd.read_csv(val_file)
            if label_col in vdf.columns:
                feat_names = [c for c in vdf.columns if c != label_col]
                up_X = vdf.drop(columns=[label_col]).values
                up_y = vdf[label_col].values
            else:
                st.warning(f"Column '{label_col}' not found. Columns: {vdf.columns.tolist()}")

        X_res, data_mode = resolve_data(ma, up_X)
        badge, desc = DATA_MODE_INFO[data_mode]
        st.info(f"**Data quality:** {badge} — {desc}")

        st.divider()
        st.subheader("Step 2 — Compatibility Check")

        if st.button("▶ Run Compatibility Check", type="primary", key="run"):
            with st.spinner("Computing ECCM…"):
                try:
                    calc     = ECCMCalculator(task=task)
                    calc.epc = epc_trainer
                    scores   = calc.compute(ma, mb, X=X_res)
                    ec       = scores["eccm"]

                    st.session_state.update({
                        "scores": scores, "ec": ec, "ma": ma, "mb": mb,
                        "X": X_res, "y": up_y, "a_n": a_name, "b_n": b_name,
                        "feat": feat_names or [], "task": task,
                        "data_mode": data_mode, "done": True,
                    })
                except Exception as e:
                    st.error(f"Computation failed: {e}")
                    st.session_state["done"] = False

        if st.session_state.get("done"):
            s  = st.session_state["scores"]
            ec = st.session_state["ec"]
            tier, colour, emoji = s["tier"], s["tier_colour"], s["tier_emoji"]
            a_n, b_n = st.session_state["a_n"], st.session_state["b_n"]

            # Tier banner
            st.markdown(
                f"<div style='padding:14px 20px;border-radius:8px;"
                f"background:{colour}22;border-left:5px solid {colour};"
                f"font-size:1.1rem;font-weight:600;'>"
                f"{emoji}&nbsp;&nbsp;{tier} — ECCM: {ec:.4f}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;Estimated success: {s['p_success']:.0%}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PSC (Weights)",   f"{s['psc']:.4f}")
            c2.metric("FSC (Behaviour)", f"{s['fsc']:.4f}")
            c3.metric("RSC (Features)",  f"{s['rsc']:.4f}")
            c4.metric("ECCM",            f"{ec:.4f}")

            w = s.get("weights", {})
            st.caption(f"**Weights [{task}]:** w_PSC={w.get('w_psc',0):.3f}  "
                       f"w_FSC={w.get('w_fsc',0):.3f}  w_RSC={w.get('w_rsc',0):.3f}")

            fig_bar = go.Figure(go.Bar(
                x=["PSC", "FSC", "RSC", "ECCM"],
                y=[s["psc"], s["fsc"], s["rsc"], ec],
                marker_color=["#4c72b0","#55a868","#c44e52","#8172b2"],
                text=[f"{v:.4f}" for v in [s["psc"], s["fsc"], s["rsc"], ec]],
                textposition="outside",
            ))
            fig_bar.update_layout(title=f"ECCM Components — {a_n} + {b_n}",
                                   yaxis=dict(range=[0,1.12]), height=310,
                                   margin=dict(t=50,b=30))
            st.plotly_chart(fig_bar, use_container_width=True)

            # Feature importance comparison
            cf1, cf2 = st.columns(2)
            with cf1:
                fig = fi_chart(st.session_state["ma"], a_n, st.session_state["feat"])
                if fig: st.plotly_chart(fig, use_container_width=True)
            with cf2:
                fig = fi_chart(st.session_state["mb"], b_n, st.session_state["feat"])
                if fig: st.plotly_chart(fig, use_container_width=True)

            # EPC evidence
            st.subheader("🔍 EPC Evidence — Nearest Historical Merges")
            rel = s.get("epc_reliability", 0.5)
            rel_badge = (
                f"🟢 High reliability ({rel:.0%})" if rel >= 0.7 else
                f"🟡 Moderate reliability ({rel:.0%})" if rel >= 0.4 else
                f"🔴 Low reliability ({rel:.0%}) — this pair is unlike past merges"
            )
            st.caption(rel_badge)
            epc_table(s.get("epc_neighbours", []))

            # XAI narrative
            st.subheader("🧠 XAI Explanation")
            st.markdown(xai_narrative(s["psc"], s["fsc"], s["rsc"], ec, a_n, b_n, task))

            # Merge step
            st.divider()
            st.subheader("Step 3 — Merge Models")

            if st.session_state["y"] is None:
                st.info("Upload a labelled CSV to evaluate merged AUC.")
            else:
                if tier == "Low Compatibility":
                    st.error("⛔ High risk — merging likely reduces performance.")
                elif tier == "Medium Compatibility":
                    st.warning("⚠️ Moderate risk — check the blend curve on the Analysis tab.")

                ok = True
                if tier in ("Low Compatibility", "Medium Compatibility"):
                    ok = st.checkbox("I understand the risk and want to merge anyway", key="ack")

                if ok:
                    blend_r  = st.slider(f"Blend weight for {a_n}", 0.0, 1.0, 0.5, 0.05)
                    if st.button("⚗️ Merge & Evaluate", type="primary", key="merge"):
                        with st.spinner("Merging…"):
                            X_m, y_m = st.session_state["X"], st.session_state["y"]
                            pa_ = st.session_state["ma"].predict_proba(X_m)[:, 1]
                            pb_ = st.session_state["mb"].predict_proba(X_m)[:, 1]
                            auc_a = roc_auc_score(y_m, pa_)
                            auc_b = roc_auc_score(y_m, pb_)
                            auc_m = roc_auc_score(y_m, blend_r * pa_ + (1-blend_r) * pb_)
                            delta = auc_m - max(auc_a, auc_b)

                        st.success(f"✅ Merged AUC: **{auc_m:.6f}**")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric(f"{a_n} AUC", f"{auc_a:.6f}")
                        mc2.metric(f"{b_n} AUC", f"{auc_b:.6f}")
                        mc3.metric("Merged AUC", f"{auc_m:.6f}", delta=f"{delta:+.6f}")

                        buf = io.BytesIO()
                        joblib.dump(BlendedModel(st.session_state["ma"],
                                                  st.session_state["mb"], blend_r), buf)
                        buf.seek(0)
                        st.download_button("⬇️ Download Merged Model", buf,
                                           f"merged_{a_n}_{b_n}_r{blend_r:.2f}.pkl",
                                           mime="application/octet-stream")
                        st.caption("The downloaded model has `feature_importances_`, `predict_proba`, "
                                   "and can be re-uploaded into this Simulator.")
    else:
        st.info("⬆️ Upload Model A and Model B to begin. Validation CSV is optional.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAIR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Pair Analysis")
    if not st.session_state.get("done"):
        st.info("Run a compatibility check on the Simulator tab first.")
    else:
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

        st.markdown(f"{emoji} {tier} — ECCM={ec:.4f} | P(success)≈{s['p_success']:.0%}")

        # Weights bar
        w = s.get("weights", {})
        if w:
            fig_w = go.Figure(go.Bar(
                x=["PSC", "FSC", "RSC"],
                y=[w["w_psc"], w["w_fsc"], w["w_rsc"]],
                marker_color=["#4c72b0","#55a868","#c44e52"],
                text=[f"{v:.3f}" for v in [w["w_psc"],w["w_fsc"],w["w_rsc"]]],
                textposition="outside",
            ))
            fig_w.update_layout(title=f"ECCM Weights [{t}]",
                                 yaxis=dict(range=[0,0.8]), height=270)
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption("Taller bar = stronger predictor of merge success for this task (answers RQ2).")

        st.divider()
        cr1, cr2 = st.columns(2)
        with cr1:
            st.subheader("ECCM Radar")
            cats = ["PSC","FSC","RSC","ECCM"]
            vals = [s["psc"], s["fsc"], s["rsc"], ec]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", fillcolor=f"{colour}33", line_color=colour,
            ))
            fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                  title=f"{a_n} + {b_n}", height=350, showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

        with cr2:
            st.subheader("Blend Ratio AUC Curve")
            if X_ is not None and y_ is not None:
                try:
                    fig_b, br, ba = blend_curve(ma_, mb_, X_, y_, a_n, b_n)
                    st.plotly_chart(fig_b, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute blend curve: {e}")
            else:
                st.info("Upload a labelled CSV to see the blend curve.")

        if X_ is not None and y_ is not None:
            st.divider()
            st.subheader("Prediction Distributions")
            try:
                pa = ma_.predict_proba(X_)[:, 1]
                pb = mb_.predict_proba(X_)[:, 1]
                fig_d = go.Figure()
                fig_d.add_trace(go.Histogram(x=pa, name=a_n, opacity=0.6,
                                              marker_color="#4c72b0", nbinsx=40))
                fig_d.add_trace(go.Histogram(x=pb, name=b_n, opacity=0.6,
                                              marker_color="#55a868", nbinsx=40))
                fig_d.update_layout(barmode="overlay", height=320,
                                     xaxis_title="P(class=1)", yaxis_title="Count")
                st.plotly_chart(fig_d, use_container_width=True)

                fig_s = px.scatter(x=pa, y=pb, opacity=0.3,
                                    labels={"x":f"{a_n} score","y":f"{b_n} score"},
                                    title="Prediction Agreement",
                                    color=y_.astype(str),
                                    color_discrete_map={"0":"#4c72b0","1":"#c44e52"})
                fig_s.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                            line=dict(dash="dash",color="grey"),
                                            name="Perfect agreement"))
                fig_s.update_layout(height=360)
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                st.warning(f"Distribution plots failed: {e}")

        st.divider()
        st.subheader("🧠 XAI Narrative")
        st.markdown(xai_narrative(s["psc"],s["fsc"],s["rsc"],ec,a_n,b_n,t))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("About Mergence")
    st.markdown("""
### ECCM

`ECCM = w_psc × PSC + w_fsc × FSC + w_rsc × RSC + w_epc × EPC`

| Metric | Description |
|--------|-------------|
| **PSC** | Parameter Space Compatibility — cosine similarity of feature importances |
| **FSC** | Functional Similarity — prediction correlation on validation data |
| **RSC** | Representational Similarity — feature importance rank correlation |
| **EPC** | Evolutionary Pressure Compatibility — k-NN contextual prediction |

Weights are **task-specific** (different for Fraud and Churn) and
derived from empirical EPC training on 276-pair merge experiments.

---

### Compatibility Tiers (data-driven thresholds)

| Tier | Fraud ECCM | Churn ECCM | Empirical P(success) |
|------|-----------|-----------|----------------------|
| ✅ High | ≥ 0.935 | ≥ 0.988 | ≥ 80% |
| ⚠️ Medium | 0.843–0.935 | 0.960–0.988 | 40–80% |
| ❌ Low | < 0.843 | < 0.960 | < 40% |

---

### Research Questions

| RQ | Where addressed |
|----|----------------|
| RQ1 — Quantify evolutionary pressure | EPC k-NN evidence table (Simulator tab) |
| RQ2 — Optimal PSC/FSC/RSC combination | ECCM weights bar chart (Analysis tab) |
| RQ3 — Efficiency over random baseline | ECCM tier gate + M2N2 optimisation results |
| RQ4 — Interpretability for non-experts | XAI narrative + EPC evidence + all chart captions |

---
*Trevin Joseph · w1953285 · BEng Software Engineering · IIT / University of Westminster · 2026*
""")