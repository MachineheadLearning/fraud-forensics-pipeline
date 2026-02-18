"""
Fraud Detection Pipeline — Dashboard
=====================================
Streamlit dashboard for fraud analysis results.
Loads only pre-computed data from dashboard_data/ — no model retraining at runtime.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dashboard_data"
STATS_PATH = DATA_DIR / "dashboard_stats.json"
RESULTS_PATH = DATA_DIR / "investigative_results.csv"

# ---------------------------------------------------------------------------
# Plotly template (consistent light look)
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", color="#1a1f2e"),
    margin=dict(l=40, r=20, t=40, b=40),
)

ACCENT = "#1E88E5"     # Strong blue
DANGER = "#E53935"     # Strong red
SUCCESS = "#43A047"    # Strong green
WARNING = "#FB8C00"    # Strong orange

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_stats() -> dict:
    with open(STATS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


# ===================================================================
# View 1 — Overview
# ===================================================================
def render_overview(stats: dict) -> None:
    st.header("Operational Overview")
    st.caption(
        "High-level detection metrics: model volume, temporal patterns, "
        "and category-level fraud exposure."
    )

    kpi = stats["kpi"]
    recall = kpi["caught_fraud"] / (kpi["caught_fraud"] + kpi["missed_fraud"])
    precision = kpi["caught_fraud"] / kpi["flagged_transactions"]

    # --- KPI row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{kpi['total_transactions']:,}")
    k2.metric("Flagged for Review", f"{kpi['flagged_transactions']:,}")
    k3.metric("Recall (Fraud Caught)", f"{recall:.1%}")
    k4.metric(
        "Transactions Saved",
        f"{kpi['total_transactions'] - kpi['flagged_transactions']:,}",
        help="Legitimate transactions that required no manual review.",
    )

    st.divider()

    # --- Hourly fraud distribution ("Unsupervised Window") ---
    st.subheader("Unsupervised Window — Hourly Fraud Distribution")
    st.caption(
        "Fraud concentrates heavily in the 22:00–03:00 window, when "
        "transaction volume drops but fraud rate spikes — a strong temporal signal "
        "captured by the model's night window feature."
    )

    # --- Night fraud KPI ---
    night_pct = stats["charts"]["night_fraud_pct"]
    nk = st.columns(1)
    nk[0].metric("Fraud in Unsupervised Window (22:00–03:00)", f"{night_pct:.1%}")

    hour_data = stats["charts"]["fraud_by_hour"]
    hours = list(range(24))
    counts = [hour_data.get(str(h), 0) for h in hours]
    labels = [f"{h:02d}:00" for h in hours]

    colors = [DANGER if c > np.percentile(counts, 75) else ACCENT for c in counts]

    fig_hour = go.Figure(
        go.Bar(x=labels, y=counts, marker_color=colors, hovertemplate="%{x}: %{y} frauds<extra></extra>")
    )
    fig_hour.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Hour of Day",
        yaxis_title="Fraudulent Transactions",
        height=350,
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # --- High-risk categories ---
    st.subheader("High-Risk Categories")
    st.caption(
        "Category-level fraud concentration. Online channels "
        "(Online Shopping, Grocery Online) dominate, consistent with "
        "card-not-present fraud patterns."
    )

    cat_data = stats["charts"]["fraud_by_category"]
    cats = list(cat_data.keys())
    vals = list(cat_data.values())

    fig_cat = go.Figure(
        go.Bar(
            y=cats[::-1],
            x=vals[::-1],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x} frauds<extra></extra>",
        )
    )
    fig_cat.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Fraud Count")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # --- Demographic Vulnerability ---
    st.subheader("Demographic Vulnerability")
    st.caption(
        "Fraud targeting follows a bimodal age distribution — not a single "
        "vulnerable cohort. The heatmap below shows fraud rate by age group and "
        "merchant category."
    )

    # --- Heatmap from pre-computed data ---
    hm = stats["demographic_heatmap"]
    hm_z = np.array(hm["values"])
    hm_text = [[f"{v:.1%}" for v in row] for row in hm_z]

    fig_hm = go.Figure(
        go.Heatmap(
            z=hm_z,
            x=hm["age_groups"],
            y=hm["categories"],
            text=hm_text,
            texttemplate="%{text}",
            textfont=dict(color="#212121"),
            colorscale=[
                [0, "#FFF5F0"],
                [0.2, "#FEE0D2"],
                [0.4, "#FCBBA1"],
                [0.6, "#FC9272"],
                [0.8, "#EF3B2C"],
                [1, "#99000D"],
            ],
            xgap=2,
            ygap=2,
            showscale=True,
            colorbar=dict(
                title=dict(text="Fraud Rate", font=dict(color="#212121")),
                tickformat=".1%",
                tickfont=dict(color="#212121")
            ),
            hovertemplate="Category: %{y}<br>Age: %{x}<br>Fraud Rate: %{text}<extra></extra>",
        )
    )
    fig_hm.update_layout(
        **PLOTLY_LAYOUT,
        height=max(500, len(hm["categories"]) * 45),
        xaxis_title="Age Group",
        yaxis_title="Category",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown(
        "The 30–50 cohort forms a clear valley — their established spending patterns make them "
        "the hardest group to exploit. Under-30s are targeted through a narrow set of digital "
        "channels (Online Shopping, Grocery In-Store), while the 50+ groups face broader and more "
        "intense exposure across more categories and at higher absolute rates, consistent with "
        "social engineering vectors."
    )
    # --- Geospatial Neutrality ---
    st.subheader("Geospatial Neutrality")
    st.markdown(
        """
        Within the dataset used in this project, fraud is uniformly distributed
        across U.S. states with no statistically significant geographic
        concentration. A detailed geospatial audit report is available [here](https://github.com/MachineheadLearning/fraud-forensics-pipeline/blob/main/notebooks/geospatial_neutrality_report.ipynb).
        """
    )


# ===================================================================
# View 2 — Transaction Auditor
# ===================================================================
def render_auditor(df: pd.DataFrame) -> None:
    st.header("Transaction Auditor")
    st.caption(
        "Filterable transaction-level view. Use the risk score and "
        "category filters to explore model predictions and identify patterns."
    )

    # --- Filters ---
    f1, f2 = st.columns([1, 2])
    with f1:
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
        )
    with f2:
        all_cats = sorted(df["category"].unique())
        selected_cats = st.multiselect(
            "Filter Categories", options=all_cats, default=all_cats
        )

    mask = (
        (df["risk_score"] >= risk_range[0])
        & (df["risk_score"] <= risk_range[1])
        & (df["category"].isin(selected_cats))
    )
    filtered = df.loc[mask].copy()

    # --- Summary metrics ---
    fraud_in_view = filtered.loc[filtered["is_fraud"] == 1]
    avg_fraud_amt = fraud_in_view["amt"].mean() if len(fraud_in_view) > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions Shown", f"{len(filtered):,}")
    m2.metric("High-Risk (>0.8)", f"{(filtered['risk_score'] > 0.8).sum():,}")
    m3.metric("Confirmed Fraud in View", f"{filtered['is_fraud'].sum():,}")
    m4.metric(
        "Avg Fraud Amount",
        f"${avg_fraud_amt:,.2f}",
        help="Mean dollar amount for confirmed-fraud transactions in the current filter.",
    )

    st.divider()

    # --- Display columns (human-readable subset) ---
    # Use raw (pre-scaling) columns for human readability
    display_cols = [
        "category", "amt", "risk_score", "is_fraud",
        "age_group_label", "spike_factor_raw", "merchant_risk", "category_risk",
        "is_night_window_raw", "velocity_burst_24h_raw",
    ]
    view_df = filtered[display_cols].copy()
    view_df = view_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Map night window to readable labels
    view_df["is_night_window_raw"] = view_df["is_night_window_raw"].map({1: "Yes", 0: "No", 1.0: "Yes", 0.0: "No"})

    # Rename for readability
    view_df.columns = [
        "Category", "Amount ($)", "Risk Score", "Actual Fraud",
        "Age Group", "Spike Factor", "Merchant Risk", "Category Risk",
        "Night Window", "Velocity Burst 24h",
    ]

    def highlight_risk(row):
        """Apply red background to high-risk rows (risk_score > 0.8)."""
        if row["Risk Score"] > 0.8:
            return ["background-color: rgba(229,57,53,0.15); color: #212121"] * len(row)
        return [""] * len(row)

    styled = (
        view_df.style
        .apply(highlight_risk, axis=1)
        .format({
            "Risk Score": "{:.4f}",
            "Amount ($)": "${:,.2f}",
            "Spike Factor": "{:.2f}",
            "Velocity Burst 24h": "{:.2f}",
        })
    )

    st.dataframe(styled, use_container_width=True, height=520)

    # --- Behavioral Indicator Key ---
    st.divider()
    st.subheader("Indicator Key")
    with st.expander("How to read the behavioural columns", expanded=True):
        st.markdown(
            """
            | Indicator | What it measures | Interpretation |
            |---|---|---|
            | **Spike Factor** | Current transaction amount ÷ customer's rolling average | Values **> 2 σ** above baseline flag spending that deviates sharply from habit — the single strongest fraud signal in the model. |
            | **Merchant Risk** | Historical fraud rate encoded per merchant | Merchants with elevated scores have disproportionate fraud histories; useful for identifying high-risk merchants. |
            | **Category Risk** | Historical fraud rate encoded per category | Captures systemic category-level exposure (e.g. `Online Shopping`). |
            | **Night Window** | Binary flag for 22:00–03:00 transactions | Encodes the "Unsupervised Window" temporal attack surface. |
            | **Velocity Burst 24 h** | Transaction frequency anomaly in a 24-hour rolling window | Rapid successive transactions suggest automated or compromised-card usage. |
            | **Age Group** | Scaled age-group indicator | Maps to the bimodal demographic pattern — extreme values correspond to Under-30 and Senior (70+) cohorts. |
            """
        )


# ===================================================================
# View 3 — Model Explainability
# ===================================================================
def render_explainability(stats: dict) -> None:
    st.header("Explainable AI (XAI)")
    st.caption(
        "Model transparency and performance documentation. "
        "This section covers the key modelling choices and their impact on results."
    )

    # --- Methodology cards ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Class Imbalance — SMOTE")
        st.markdown(
            "**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic "
            "fraud examples via k-nearest-neighbour interpolation to counter the severe "
            "≈0.58% fraud rate. It was applied **exclusively to the training fold** — the "
            "test set retains the real-world class distribution, so all evaluation metrics "
            "are production-representative."
        )

    with c2:
        st.subheader("Feature Transparency — SHAP")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) decomposes every prediction into "
            "per-feature contributions, making the XGBoost ensemble explainable. "
            "The top three risk drivers are **transaction amount** (the single strongest "
            "predictor), **category risk** (target-encoded fraud rate per merchant category), "
            "and the **night window flag** (22:00–03:00), which encodes the temporal attack "
            "surface visible in the Overview."
        )

    st.divider()

    # --- SMOTE Technicality Card ---
    st.subheader("Why Balance Only the Training Set?")
    with st.expander("Technical detail — SMOTE scope", expanded=False):
        st.markdown(
            """
            Applying SMOTE to the **full dataset** before splitting would leak
            synthetic minority samples into the test fold, producing artificially
            inflated recall and precision scores. By restricting SMOTE to the
            **training partition only**, we ensure:

            1. **Evaluation integrity** — the test set retains the real-world
               **0.39 % fraud prevalence**, so every metric (recall, precision,
               PR-AUC) reflects production conditions.
            2. **No information leakage** — synthetic samples are generated from
               training-fold neighbours only; no test-fold data points influence
               the oversampling.
            3. **Threshold calibration** — because the test distribution is
               unaltered, the chosen classification threshold transfers directly
               to deployment without recalibration.
            """
        )

    st.divider()

    # --- SHAP Global Feature Importance ---
    st.subheader("Global Feature Impact (SHAP)")
    st.caption(
        "SHAP values quantify each feature's average contribution to the model's "
        "predictions. Higher values indicate greater influence on fraud detection decisions."
    )

    shap_data = stats["shap_importance"]
    shap_features = list(shap_data.keys())
    shap_values_list = list(shap_data.values())

    fig_shap = go.Figure(
        go.Bar(
            y=shap_features[::-1],
            x=shap_values_list[::-1],
            orientation="h",
            marker_color=ACCENT,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig_shap.update_layout(
        **PLOTLY_LAYOUT,
        height=max(380, len(shap_features) * 28),
        xaxis_title="mean(|SHAP value|)",
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()

    # --- Advisory Lift ---
    st.subheader("Advisory Lift over Baseline")

    lift_data = stats["advisory_lift"]
    baseline_missed = lift_data["baseline_missed"]
    model_missed = lift_data["model_missed"]
    saved = lift_data["transactions_saved"]
    model_caught = lift_data["model_caught"]
    reduction_pct = (saved / baseline_missed) * 100

    l1, l2, l3 = st.columns(3)
    l1.metric("Baseline Missed (Dummy)", f"{baseline_missed:,} cases")
    l2.metric("Model Missed (XGBoost)", f"{model_missed:,} cases")
    l3.metric("Reduction", f"{reduction_pct:.0f}%", help="Percentage reduction in missed fraud vs. stratified baseline.")

    st.success(
        f"The champion model reduced missed fraud from **{baseline_missed:,}** cases "
        f"(stratified baseline) to **{model_missed:,}** — a **{reduction_pct:.0f}% reduction**. "
        f"This translates to **{saved:,} additional fraud cases caught** that would have "
        f"been missed under random classification, with the model successfully identifying "
        f"**{model_caught:,}** total fraud cases."
    )

    st.divider()

    # --- Confusion Matrix ---
    st.subheader("Model Performance — Confusion Matrix")
    cm = stats["performance"]["cm"]

    cm_styles = {
        "TN": ("Legitimate (TN)", cm["tn"], "#E3F2FD", "#1a1f2e"),
        "FP": ("False Alarm (FP)", cm["fp"], "#FFF3E0", "#1a1f2e"),
        "FN": ("Missed Fraud (FN)", cm["fn"], "#E53935", "#FFFFFF"),
        "TP": ("Caught Fraud (TP)", cm["tp"], "#43A047", "#FFFFFF"),
    }

    row1_a, row1_b = st.columns(2)
    row2_a, row2_b = st.columns(2)

    for col, key in zip([row1_a, row1_b, row2_a, row2_b], ["TN", "FP", "FN", "TP"]):
        label, count, bg, fg = cm_styles[key]
        with col:
            st.markdown(
                f'<div style="background:{bg};border-radius:8px;padding:18px 14px;text-align:center;">'
                f'<span style="font-size:0.9rem;color:{fg};opacity:0.8;">{label}</span><br>'
                f'<span style="font-size:1.8rem;font-weight:700;color:{fg};">{count:,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Derived metrics ---
    recall = cm["tp"] / (cm["tp"] + cm["fn"])
    precision = cm["tp"] / (cm["tp"] + cm["fp"])
    f1 = 2 * precision * recall / (precision + recall)
    specificity = cm["tn"] / (cm["tn"] + cm["fp"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{recall:.1%}")
    m2.metric("Precision", f"{precision:.1%}")
    m3.metric("F1 Score", f"{f1:.3f}")
    m4.metric("Specificity", f"{specificity:.4f}")

    st.divider()

    # --- Precision-Recall Curve ---
    st.subheader("Precision-Recall Curve")
    st.caption(
        "In fraud detection, the PR curve is more informative than "
        "ROC because it is sensitive to the minority class. The shaded area represents "
        "the operating region where recall exceeds 90 %."
    )

    pr = stats["performance"]["pr_curve"]

    fig_pr = go.Figure()
    fig_pr.add_trace(
        go.Scatter(
            x=pr["recall"],
            y=pr["precision"],
            mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(30,136,229,0.12)",
            name="PR Curve",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )
    fig_pr.add_trace(
        go.Scatter(
            x=[recall],
            y=[precision],
            mode="markers",
            marker=dict(size=12, color="gold", line=dict(width=2, color="black")),
            name="Operating Point (t=0.5)",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )
    fig_pr.add_hline(
        y=precision, line_dash="dot", line_color=WARNING,
        annotation_text=f"Operating Point (P={precision:.2%})",
        annotation_font_color=WARNING,
    )
    fig_pr.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=380,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # --- Threshold Sensitivity ---
    st.subheader("Threshold Sensitivity Analysis")
    st.caption(
        "How model performance changes across classification thresholds. "
        "The selected operating point (t=0.50) is highlighted."
    )

    sweep_data = stats["threshold_sweep"]
    sweep_df = pd.DataFrame(sweep_data)
    sweep_df.columns = ["Threshold", "Recall", "Precision", "Flagged"]
    sweep_df["Threshold"] = sweep_df["Threshold"].map("{:.2f}".format)
    sweep_df["Recall"] = sweep_df["Recall"].map("{:.2%}".format)
    sweep_df["Precision"] = sweep_df["Precision"].map("{:.2%}".format)
    sweep_df["Flagged"] = sweep_df["Flagged"].map("{:,}".format)

    def highlight_operating_point(row):
        if row["Threshold"] == "0.50":
            return ["background-color: rgba(30,136,229,0.18); color: #212121; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_sweep = sweep_df.style.apply(highlight_operating_point, axis=1)
    st.dataframe(styled_sweep, use_container_width=True, hide_index=True)


# ===================================================================
# View 4 — Residual Risk Audit
# ===================================================================
def render_roadmap(stats: dict) -> None:
    st.header("Residual Risk Audit")
    st.caption(
        "Profiling the model's blind spots — what it misses, why, "
        "and the residual risk characteristics."
    )

    # --- The Stealth Boundary ---
    blind = stats["blind_spots"]
    missed = blind["total_missed"]

    st.subheader(f"The Stealth Boundary — {missed} Missed Cases")

    st.markdown(
        f"""
        The champion model missed **{missed}** fraudulent transactions (false
        negatives) with a median transaction amount of **${blind['median_missed_amt']:.2f}**.
        These cases represent transactions whose amounts and frequencies are
        indistinguishable from legitimate behaviour — the model's current
        detection ceiling.
        """
    )

    # Stealth boundary visualisation — driven by stats, not hardcoded
    missed_by_cat = blind["by_category"]
    cat_names = list(missed_by_cat.keys())
    cat_counts = list(missed_by_cat.values())

    # Colour the top 3 categories in DANGER, rest in WARNING
    sorted_pairs = sorted(zip(cat_names, cat_counts), key=lambda x: x[1], reverse=True)
    top3 = {name for name, _ in sorted_pairs[:3]}
    chart_names = [p[0] for p in sorted_pairs][::-1]
    chart_vals = [p[1] for p in sorted_pairs][::-1]
    chart_colors = [DANGER if n in top3 else WARNING for n in chart_names]

    fig_stealth = go.Figure(
        go.Bar(
            y=chart_names,
            x=chart_vals,
            orientation="h",
            marker_color=chart_colors,
            hovertemplate="%{y}: %{x} missed<extra></extra>",
        )
    )
    fig_stealth.update_layout(
        **PLOTLY_LAYOUT,
        height=max(320, len(cat_names) * 30),
        xaxis_title="Missed Fraud Cases (FN)",
        title=dict(text="False Negatives by Category", font_size=14),
    )
    st.plotly_chart(fig_stealth, use_container_width=True)

    st.warning(
        f"**Stealth Boundary:** The missed fraud cases span {len(cat_names)} categories. "
        f"The top category alone accounts for {sorted_pairs[0][1]} missed cases "
        f"({sorted_pairs[0][1]/missed:.0%} of all false negatives), indicating concentrated "
        f"residual risk in categories where fraud mimics legitimate behaviour."
    )

    st.divider()

    # --- Risk Band Distribution ---
    st.subheader("Risk Band Distribution of Missed Cases")
    st.caption(
        "Category risk was bucketed into quintiles — this shows where missed "
        "fraud falls across the risk spectrum."
    )

    risk_band_data = blind["by_risk_band"]
    # Order logically, not alphabetically
    band_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    band_counts = [risk_band_data.get(b, 0) for b in band_order]
    band_colors = [ACCENT, "#66BB6A", WARNING, "#EF7350", DANGER]

    fig_bands = go.Figure(
        go.Bar(
            y=band_order,
            x=band_counts,
            orientation="h",
            marker_color=band_colors,
            hovertemplate="%{y}: %{x} missed<extra></extra>",
        )
    )
    fig_bands.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis_title="Missed Fraud Cases",
    )
    st.plotly_chart(fig_bands, use_container_width=True)

    total_missed_bands = sum(band_counts)
    high_pct = risk_band_data.get("High", 0) / total_missed_bands * 100 if total_missed_bands else 0
    vlow_pct = risk_band_data.get("Very Low", 0) / total_missed_bands * 100 if total_missed_bands else 0
    st.markdown(
        f"Missed fraud is **not** concentrated in low-risk categories alone. The roughly "
        f"even split between High ({high_pct:.0f}%) and Very Low ({vlow_pct:.0f}%) risk bands "
        f"means the model's blind spot spans the full risk spectrum — what unifies the missed "
        f"cases is their low dollar amount, not their category risk profile. This suggests "
        f"the stealth boundary is primarily an amount-based limitation, not a category-based one."
    )


# ===================================================================
# Sidebar
# ===================================================================
def render_sidebar() -> str:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/detective.png",
            width=64,
        )
        st.title("Fraud Detection Dashboard")
        st.caption("Analytics & Model Performance")

        st.divider()

        view = st.radio(
            "Navigation",
            options=["Overview", "Transaction Auditor", "Model Explainability", "Residual Risk Audit"],
            index=0,
        )

        st.divider()

        with st.expander("Project Methodology", expanded=False):
            st.markdown(
                """
                **Architecture**
                - Selected model: **XGBoost** (gradient-boosted ensemble)
                - Imbalance handling: **SMOTE** on training data only
                - Feature engineering: temporal, behavioural, and demographic signals
                - Threshold: **t=0.50** (89.7% recall, 52.7% precision)

                **Data**
                - Sparkov synthetic dataset (Kaggle)
                - 555 k+ test-set transactions evaluated
                - Pre-computed scores — no live inference

                **Stack**
                - Python · scikit-learn · XGBoost
                - Streamlit · Plotly
                """
            )

        st.divider()
        st.caption("All data is synthetic — findings demonstrate methodology only.")

    return view


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    st.set_page_config(
        page_title="AI-Powered Fraud Detection Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load data once
    stats = load_stats()
    results = load_results()

    # Sidebar navigation
    view = render_sidebar()

    # Route to selected view
    if view == "Overview":
        render_overview(stats)
    elif view == "Transaction Auditor":
        render_auditor(results)
    elif view == "Model Explainability":
        render_explainability(stats)
    elif view == "Residual Risk Audit":
        render_roadmap(stats)


if __name__ == "__main__":
    main()
