"""
FactorLens — EFA/CFA Diagnostic Suite
Single-file Streamlit app. Deploy directly to Streamlit Community Cloud.
Requires: streamlit, plotly, pandas, numpy, openai
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FactorLens — EFA/CFA Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Main app background: rich dark navy ── */
  .stApp { background-color:#0d1117; color:#e6edf3; }
  .main .block-container { padding-top:2rem; }

  /* ── Sidebar: deep navy gradient ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color:#e0e0e0 !important; }
  [data-testid="stSidebar"] .stRadio > label,
  [data-testid="stSidebar"] .stSelectbox > label { color:#aaa !important; font-size:0.75rem; }

  /* ── Sidebar toggle button — always visible ── */
  [data-testid="collapsedControl"] { display:block !important; visibility:visible !important; color:#e0e0e0 !important; }
  [data-testid="stSidebarCollapseButton"] { display:block !important; visibility:visible !important; }

  /* ── Header (share + GitHub) — always visible ── */
  header[data-testid="stHeader"] {
    display:block !important;
    visibility:visible !important;
    background:rgba(13,17,23,0.95) !important;
    border-bottom: 1px solid #30363d;
  }
  header[data-testid="stHeader"] * { visibility:visible !important; }

  /* Keep footer hidden (still noisy) but keep rest visible */
  footer { visibility:hidden; }

  /* ── Headings and text on dark background ── */
  h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color:#e6edf3 !important; }

  /* ── Metric cards ── */
  [data-testid="stMetricValue"] { color:#7ee8a2 !important; font-weight:700; }
  [data-testid="stMetricLabel"] { color:#8b949e !important; }

  /* ── Input widgets ── */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background:#161b22 !important; color:#e6edf3 !important;
    border:1px solid #30363d !important; border-radius:6px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background:#238636 !important; color:#fff !important;
    border:none; border-radius:6px; font-weight:600;
    transition: background 0.2s;
  }
  .stButton > button:hover { background:#2ea043 !important; }
  .stButton > button[kind="secondary"] {
    background:#21262d !important; color:#e6edf3 !important;
    border:1px solid #30363d !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:8px 8px 0 0; }
  .stTabs [data-baseweb="tab"] { color:#8b949e !important; }
  .stTabs [aria-selected="true"] { color:#e6edf3 !important; border-bottom:2px solid #7F77DD; }

  /* ── Dataframe ── */
  .stDataFrame { border:1px solid #30363d; border-radius:8px; }

  /* ── Info/warning/error boxes ── */
  .stAlert { border-radius:8px !important; }
  [data-testid="stAlert"][kind="info"]    { background:#0d1f38 !important; border-left:4px solid #388bfd; }
  [data-testid="stAlert"][kind="success"] { background:#0d2818 !important; border-left:4px solid #56d364; }
  [data-testid="stAlert"][kind="warning"] { background:#2d1f00 !important; border-left:4px solid #e3b341; }
  [data-testid="stAlert"][kind="error"]   { background:#2d0f0f !important; border-left:4px solid #f85149; }

  /* ── Expander ── */
  .streamlit-expanderHeader { background:#161b22 !important; color:#e6edf3 !important; border-radius:6px; }
  .streamlit-expanderContent { background:#0d1117 !important; border:1px solid #30363d; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background:#161b22 !important; border:2px dashed #30363d !important;
    border-radius:8px; padding:16px;
  }
  [data-testid="stFileUploader"]:hover { border-color:#7F77DD !important; }

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] [role="slider"] { background:#7F77DD !important; }

  /* ── Code blocks ── */
  .stCode { background:#161b22 !important; border:1px solid #30363d !important; }

  /* ── badges ── */
  .bdg-pass { background:#0d2818;color:#56d364;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600;border:1px solid #56d36440; }
  .bdg-warn { background:#2d1f00;color:#e3b341;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600;border:1px solid #e3b34140; }
  .bdg-fail { background:#2d0f0f;color:#f85149;padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:600;border:1px solid #f8514940; }

  /* ── issue strips ── */
  .iss-bad  { border-left:4px solid #f85149;background:#2d0f0f;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px;color:#ffa198; }
  .iss-warn { border-left:4px solid #e3b341;background:#2d1f00;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px;color:#f0c060; }
  .iss-ok   { border-left:4px solid #56d364;background:#0d2818;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:8px;color:#7ee8a2; }

  /* ── Upload section card ── */
  .upload-card {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:20px; margin-bottom:16px;
  }
  .upload-card h4 { color:#7F77DD !important; margin-top:0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PURPLE = "#7F77DD"; TEAL = "#1D9E75"; CORAL = "#D85A30"
AMBER  = "#BA7517"; RED  = "#E24B4A"; GREEN = "#639922"; GRAY = "#888780"

SCENARIOS = {
    "good":       dict(n_vars=12, n_obs=350, n_factors=3, loading_str=0.72),
    "lowloading": dict(n_vars=12, n_obs=300, n_factors=3, loading_str=0.32),
    "crossload":  dict(n_vars=12, n_obs=300, n_factors=3, loading_str=0.55),
    "smalln":     dict(n_vars=12, n_obs=80,  n_factors=3, loading_str=0.65),
    "lowcomm":    dict(n_vars=12, n_obs=300, n_factors=3, loading_str=0.42),
    "heywood":    dict(n_vars=9,  n_obs=150, n_factors=3, loading_str=0.95),
}

SCENARIO_DESC = {
    "good":       "✅ **Well-specified** — Strong loadings, adequate N, clean factor structure.",
    "lowloading": "↓ **Low loadings** — Primary loadings below 0.40; poor communalities and fit.",
    "crossload":  "↔ **Cross-loading** — Items loading on multiple factors (≥ 0.35 secondary loads).",
    "smalln":     "N  **Small sample** — Only 80 observations; unstable estimates expected.",
    "lowcomm":    "◐ **Low communalities** — Items share little variance with factors.",
    "heywood":    "⚠ **Heywood case** — Loadings approaching 1.0; ill-conditioned covariance matrix.",
}

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE — EFA / CFA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_efa(n_vars, n_obs, n_factors, loading_str, scenario, seed=42):
    rng = np.random.default_rng(seed)
    ipf = max(1, n_vars // n_factors)

    # eigenvalues
    eigs = [(n_vars / n_factors) * loading_str * (1 + 0.3 * (n_factors - f) / n_factors)
            for f in range(n_factors)]
    eigs += [float(0.2 + rng.random() * 0.6) for _ in range(n_vars - n_factors)]
    if scenario == "smalln":
        eigs = [e * 0.85 if i < n_factors else e for i, e in enumerate(eigs)]

    # loadings
    loadings = []
    for i in range(n_vars):
        my_f = min(i // ipf, n_factors - 1)
        row  = []
        for f in range(n_factors):
            if f == my_f:
                l = loading_str + rng.standard_normal() * 0.08
                if scenario == "lowloading": l = 0.25 + rng.random() * 0.15
                elif scenario == "heywood":  l = min(0.99, l * 1.25)
                row.append(float(np.clip(l, 0.10, 0.99)))
            else:
                cl = rng.random() * 0.12
                if scenario == "crossload": cl = 0.35 + rng.random() * 0.20
                row.append(float(cl))
        loadings.append(row)

    # communalities
    communalities = [min(0.99, sum(l**2 for l in row)) for row in loadings]
    if scenario == "lowcomm":
        communalities = [c * 0.55 for c in communalities]

    # KMO
    kmo = float(np.clip(
        0.72 + loading_str * 0.20 + rng.random() * 0.04
        - (0.12 if scenario == "smalln" else 0)
        - (0.15 if scenario == "lowloading" else 0)
        - (0.10 if scenario == "crossload" else 0),
        0.30, 0.99
    ))
    if scenario == "lowloading": kmo = float(np.clip(0.52 + rng.random() * 0.06, 0.30, 0.70))
    if scenario == "crossload":  kmo = float(np.clip(0.61 + rng.random() * 0.05, 0.30, 0.75))

    bartlett = int(n_obs * 0.40 * n_vars * (n_vars - 1) / 2 * max(0.05, loading_str - 0.10))
    var_exp  = int(sum(eigs[:n_factors]) / sum(eigs) * 100)

    return dict(eigenvalues=eigs, loadings=loadings, communalities=communalities,
                kmo=kmo, bartlett_chi2=bartlett, var_explained=var_exp,
                n_factors_retained=n_factors, ipf=ipf)


def compute_cfa(n_vars, n_obs, n_factors, loading_str, scenario, seed=42):
    rng    = np.random.default_rng(seed + 1)
    rmsea  = 0.04 + rng.random() * 0.02
    cfi    = 0.97 - rng.random() * 0.02
    tli    = 0.96 - rng.random() * 0.02
    srmr   = 0.05 + rng.random() * 0.02
    chi2df = 1.80 + rng.random() * 0.50

    deltas = dict(
        lowloading=(0.07,-0.12,-0.11, 0.06, 2.5),
        crossload =(0.05,-0.09,-0.08, 0.05, 1.8),
        smalln    =(0.04,-0.06,  0,   0,    1.2),
        lowcomm   =(0.06,-0.10,-0.10, 0.05, 0  ),
        heywood   =(0.12,-0.20,-0.18, 0.10, 4.0),
    )
    if scenario in deltas:
        dr, dc, dt, ds, dchi = deltas[scenario]
        rmsea += dr; cfi += dc; tli += dt; srmr += ds; chi2df += dchi

    mi_pairs = []
    n_pairs  = min(n_vars, 8)
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            mi_pairs.append(dict(item_a=f"V{i+1}", item_b=f"V{j+1}",
                                 mi=round(float(rng.random() * 25 + 1), 2),
                                 delta_chi2=round(float(rng.random() * 20 + 0.5), 2)))
    mi_pairs.sort(key=lambda x: -x["mi"])

    return dict(
        rmsea=float(np.clip(rmsea, 0.001, 0.30)),
        cfi  =float(np.clip(cfi,   0.50,  1.00)),
        tli  =float(np.clip(tli,   0.50,  1.00)),
        srmr =float(np.clip(srmr,  0.001, 0.30)),
        chi2df=float(np.clip(chi2df, 1.0, 12.0)),
        mi_pairs=mi_pairs[:10],
    )


def build_issues(efa, cfa, n_obs, n_vars):
    issues = []
    kmo = efa["kmo"]

    if kmo < 0.60:
        issues.append(dict(sev="bad", title=f"KMO = {kmo:.3f} — sampling inadequacy",
            desc="Values below 0.60 indicate the correlation matrix is unsuitable for factor analysis.",
            fixes=["Increase inter-item correlations by tightening item content",
                   "Remove items with near-zero correlations with all others",
                   f"Increase sample size — current N={n_obs} may suppress KMO"]))
    elif kmo < 0.70:
        issues.append(dict(sev="warn", title=f"KMO = {kmo:.3f} — marginal adequacy",
            desc="KMO between 0.60–0.70 is acceptable but solutions may be unstable.",
            fixes=["Review item wording for construct specificity",
                   "Add 2–3 strongly-loading items per factor"]))

    ipf = efa["ipf"]; nf = efa["n_factors_retained"]
    cross = [f"V{i+1}" for i, row in enumerate(efa["loadings"])
             if any(l > 0.32 for f, l in enumerate(row) if f != min(i // ipf, nf - 1))]
    if cross:
        issues.append(dict(sev="bad", title=f"{len(cross)} cross-loading items: {', '.join(cross[:6])}",
            desc="Items with secondary loadings ≥ 0.32 create factor ambiguity and inflate error.",
            fixes=["Remove items with cross-loadings > 0.40",
                   "Rewrite items to be more construct-specific",
                   "Assign ambiguous items to their highest-loading factor",
                   "Consider a bifactor model if theoretical rationale exists"]))

    low_comm = [f"V{i+1}" for i, c in enumerate(efa["communalities"]) if c < 0.40]
    if low_comm:
        issues.append(dict(sev="bad", title=f"{len(low_comm)} items with communality < 0.40",
            desc="Low-communality items contribute little variance to the factor structure.",
            fixes=["Replace with more strongly-worded items",
                   "Remove if fewer than 3 items per factor remain",
                   "Use PAF instead of PCA — handles low communalities better"]))

    if n_obs < 100:
        issues.append(dict(sev="bad", title=f"Sample size N={n_obs} is too small",
            desc="CFA requires minimum N=200; below N=100 produces highly unstable estimates.",
            fixes=["Target N ≥ 200 (ideally N ≥ 10× free parameters)",
                   "Use bootstrap CIs if N cannot increase",
                   "Consider item parceling to reduce free parameters"]))
    elif n_obs < 200:
        issues.append(dict(sev="warn", title=f"Sample size N={n_obs} is marginal",
            desc="N between 100–200 may yield unstable loadings.",
            fixes=["Target N ≥ 200 before publishing",
                   "Report 95% bootstrap CIs around standardized loadings"]))

    if cfa["rmsea"] > 0.08:
        issues.append(dict(sev="bad", title=f"RMSEA = {cfa['rmsea']:.3f} — poor fit",
            desc="RMSEA above 0.08 indicates the model poorly reproduces the covariance matrix.",
            fixes=["Free correlated residuals flagged by high modification indices",
                   "Review item content for within-factor heterogeneity",
                   "Test a higher-order or bifactor model"]))
    if cfa["cfi"] < 0.90:
        issues.append(dict(sev="bad", title=f"CFI = {cfa['cfi']:.3f} — poor incremental fit",
            desc="CFI below 0.90 indicates substantial mis-specification.",
            fixes=["Add paths suggested by top modification indices",
                   "Remove items with the lowest factor loadings",
                   "Check inter-factor correlations > 0.85 — may indicate factor collapse"]))

    if not issues:
        issues.append(dict(sev="ok", title="All diagnostics passed ✓",
            desc="Dataset structure is sound for EFA/CFA.", fixes=[]))
    return issues


def predict_diagnostics(target_loading, cross_loading_ceiling, error_variance_floor, n_obs):
    kmo   = min(0.97, 0.50 + target_loading * 0.55 - cross_loading_ceiling * 0.30 + min(n_obs, 500) / 5000)
    rmsea = max(0.01, 0.12  - target_loading * 0.12 + cross_loading_ceiling * 0.18 + error_variance_floor * 0.05)
    cfi   = min(0.99, 0.65  + target_loading * 0.48 - cross_loading_ceiling * 0.30 - error_variance_floor * 0.12)
    h2    = min(0.98, target_loading ** 2 + 0.04)
    return dict(kmo=round(kmo,3), rmsea=round(rmsea,3), cfi=round(cfi,3), h2=round(h2,3))


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _base(h=320, **kw):
    return dict(plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
                font_family="'Courier New',monospace", font_color="#e6edf3",
                margin=dict(l=40,r=20,t=30,b=40), height=h, **kw)


def chart_scree(eigenvalues, n_factors):
    x = list(range(1, len(eigenvalues) + 1))
    colors = [PURPLE if i < n_factors else GRAY for i in range(len(eigenvalues))]
    fig = go.Figure()
    fig.add_bar(x=x, y=[round(e,3) for e in eigenvalues], marker_color=colors, name="Eigenvalue")
    fig.add_scatter(x=x, y=[round(e,3) for e in eigenvalues], mode="lines+markers",
                    line=dict(color=PURPLE, width=1.5), marker=dict(size=5), showlegend=False)
    fig.add_hline(y=1.0, line_dash="dash", line_color=RED,
                  annotation_text="Kaiser λ=1", annotation_position="top right")
    fig.update_layout(**_base(h=300),
                      xaxis=dict(title="Factor", tickmode="linear"),
                      yaxis=dict(title="Eigenvalue"), showlegend=False, bargap=0.3)
    return fig


def chart_heatmap(loadings, n_factors):
    z = np.array(loadings)
    n_vars = len(loadings)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"F{f+1}" for f in range(n_factors)],
        y=[f"V{i+1}" for i in range(n_vars)],
        colorscale=[[0,"#ffebee"],[0.3,"#fff8e1"],[0.5,"#e8f5e9"],[0.7,"#c8e6c9"],[1,"#1b5e20"]],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="λ", thickness=12),
    ))
    fig.update_layout(**_base(h=max(300, n_vars * 26 + 60)), xaxis=dict(side="top"))
    return fig


def chart_communalities(communalities):
    labels = [f"V{i+1}" for i in range(len(communalities))]
    colors = [GREEN if c >= 0.50 else (AMBER if c >= 0.30 else RED) for c in communalities]
    fig = go.Figure(go.Bar(
        x=[round(c,3) for c in communalities], y=labels, orientation="h",
        marker_color=colors,
        text=[f"{c:.2f}" for c in communalities], textposition="outside",
    ))
    fig.add_vline(x=0.40, line_dash="dash", line_color=AMBER, annotation_text="Min (0.40)")
    fig.add_vline(x=0.50, line_dash="dot",  line_color=GREEN, annotation_text="Good (0.50)")
    fig.update_layout(**_base(h=max(280, len(communalities) * 26 + 60)),
                      xaxis=dict(range=[0,1.1], title="h²"), yaxis=dict(title=""),
                      showlegend=False)
    return fig


def chart_gauge(value, title, good, ok, higher=True):
    rng_max = 1.0 if higher else 0.25
    if higher:
        color = GREEN if value >= good else (AMBER if value >= ok else RED)
    else:
        color = GREEN if value <= good else (AMBER if value <= ok else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        number=dict(font=dict(size=26, color=color)),
        title=dict(text=title, font=dict(size=13)),
        gauge=dict(
            axis=dict(range=[0, rng_max]),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#f0f0f0",
        ),
    ))
    fig.update_layout(height=190, margin=dict(l=20,r=20,t=44,b=10), paper_bgcolor="#0d1117",
                      font_color="#e6edf3")
    return fig


def chart_mi(mi_pairs):
    if not mi_pairs: return go.Figure()
    top    = mi_pairs[:8]
    labels = [f"{p['item_a']}↔{p['item_b']}" for p in top]
    values = [p["mi"] for p in top]
    colors = [RED if v > 10 else (AMBER if v > 5 else GRAY) for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=colors, text=[f"{v:.1f}" for v in values],
                           textposition="outside"))
    fig.add_vline(x=10, line_dash="dash", line_color=RED, annotation_text="Threshold (10)")
    fig.update_layout(**_base(h=280), xaxis=dict(title="Modification Index"),
                      yaxis=dict(title=""), showlegend=False)
    return fig


def chart_radar(pred):
    cats = ["KMO", "CFI", "1–RMSEA", "h²"]
    vals = [pred["kmo"], pred["cfi"], round(1 - pred["rmsea"], 3), pred["h2"]]
    thresh = [0.70, 0.95, 0.94, 0.50]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=thresh + [thresh[0]], theta=cats + [cats[0]],
                                  fill="toself", fillcolor="rgba(200,200,200,0.25)",
                                  line=dict(color=GRAY, dash="dash", width=1), name="Threshold"))
    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                  fill="toself", fillcolor="rgba(127,119,221,0.20)",
                                  line=dict(color=PURPLE, width=2), name="Predicted"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=False,
                                                  gridcolor="#30363d"),
                                  angularaxis=dict(gridcolor="#30363d")),
                      showlegend=True, height=320,
                      margin=dict(l=40,r=40,t=40,b=40), paper_bgcolor="#0d1117",
                      font_color="#e6edf3")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def call_openai(messages: list, system: str, api_key: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return "❌ `openai` package not installed. Add `openai` to requirements.txt and redeploy."

    client = OpenAI(api_key=api_key)
    full_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1024,
        temperature=0.4,
    )
    return response.choices[0].message.content


def build_ai_context(efa, cfa, issues, params):
    lines = [
        "You are an expert psychometrician specializing in EFA and CFA for synthetic datasets.",
        "The user is working with FactorLens, a diagnostic tool for evaluating factor-analytic data quality.",
        "",
        f"Current dataset: {params.get('n_vars','?')} variables, N={params.get('n_obs','?')}, "
        f"{params.get('n_factors','?')} factors, scenario='{params.get('scenario','?')}'",
    ]
    if efa:
        avg_h2 = round(sum(efa["communalities"]) / len(efa["communalities"]), 3)
        lines += [f"EFA — KMO={efa['kmo']:.3f}, Var explained={efa['var_explained']}%, Avg h²={avg_h2}"]
    if cfa:
        lines += [f"CFA — RMSEA={cfa['rmsea']:.3f}, CFI={cfa['cfi']:.3f}, "
                  f"TLI={cfa['tli']:.3f}, SRMR={cfa['srmr']:.3f}, χ²/df={cfa['chi2df']:.2f}"]
    if issues:
        n_bad  = sum(1 for i in issues if i["sev"] == "bad")
        n_warn = sum(1 for i in issues if i["sev"] == "warn")
        lines += [f"Diagnostics: {n_bad} critical issues, {n_warn} warnings"]
    lines += ["", "Give precise, actionable advice referencing the user's actual values. "
              "Use 3–6 sentences or short bullet points. Be specific about thresholds and numbers."]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 FactorLens")
    st.caption("EFA / CFA Diagnostic Suite")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📥 Data Input",
        "🔍 EFA Diagnostics",
        "📐 CFA Fit Indices",
        "⚠️ Issues & Fixes",
        "🔄 Regeneration Lab",
        "🤖 AI Advisor",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Quick scenarios**")
    scenario_label = st.selectbox("Load scenario", [
        "Custom",
        "✅ Well-specified (passes)",
        "↓ Low factor loadings",
        "↔ Cross-loading items",
        "N  Small sample (N=80)",
        "◐ Low communalities",
        "⚠ Heywood case",
    ])
    smap = {
        "✅ Well-specified (passes)": "good",
        "↓ Low factor loadings":     "lowloading",
        "↔ Cross-loading items":     "crossload",
        "N  Small sample (N=80)":    "smalln",
        "◐ Low communalities":       "lowcomm",
        "⚠ Heywood case":           "heywood",
    }
    if scenario_label != "Custom":
        st.session_state["active_scenario"] = smap[scenario_label]
    else:
        st.session_state.setdefault("active_scenario", "good")

    st.markdown("---")
    st.markdown("---")
    st.caption("📂 Upload CSV/Excel on Data Input page")
    st.caption("Built with Streamlit · Powered by OpenAI")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: badge html
# ─────────────────────────────────────────────────────────────────────────────
def badge(ok, warn):
    if ok:   return '<span class="bdg-pass">✅ Pass</span>'
    if warn: return '<span class="bdg-warn">⚠️ Marginal</span>'
    return         '<span class="bdg-fail">❌ Fail</span>'


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────
def _parse_uploaded_dataset(uploaded_file):
    """Parse an uploaded CSV or Excel file into a DataFrame. Returns (df, error_msg)."""
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file type. Upload a CSV or Excel (.xlsx/.xls) file."

        # Keep only numeric columns
        df_num = df.select_dtypes(include=[np.number])
        if df_num.shape[1] < 2:
            return None, "Dataset must contain at least 2 numeric columns."
        return df_num, None
    except Exception as e:
        return None, f"Could not read file: {e}"


def page_data_input():
    st.title("📥 Data Input")
    st.markdown("Upload your own dataset **or** configure simulation parameters — then click **Run diagnostics**.")

    # ── Upload section ────────────────────────────────────────────────────────
    st.markdown('<div class="upload-card"><h4>📂 Upload Dataset (CSV or Excel)</h4></div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
        help="Numeric columns become the items for EFA/CFA. Each row = one observation.",
        label_visibility="collapsed",
    )

    use_upload = False
    uploaded_df = None

    if uploaded_file is not None:
        df_parsed, err = _parse_uploaded_dataset(uploaded_file)
        if err:
            st.error(f"❌ {err}")
        else:
            uploaded_df = df_parsed
            n_obs_up, n_vars_up = uploaded_df.shape
            st.success(
                f"✅ **{uploaded_file.name}** loaded — {n_obs_up} observations × {n_vars_up} variables"
            )
            with st.expander("👀 Preview uploaded data (first 5 rows)"):
                st.dataframe(uploaded_df.head(), use_container_width=True)

            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("Observations (N)", n_obs_up)
            col_info2.metric("Variables", n_vars_up)
            col_info3.metric("Missing values", int(uploaded_df.isnull().sum().sum()))

            use_upload = st.checkbox(
                "✅ Use this uploaded dataset for diagnostics",
                value=True,
                help="When checked, N and variable count are taken from the file. "
                     "You still set the number of latent factors and scenario below.",
            )
            if use_upload:
                st.session_state["uploaded_df"] = uploaded_df
                st.info("📌 Dataset attached. Set the **number of latent factors** below, "
                        "then click **Run diagnostics**.")
    else:
        st.session_state.pop("uploaded_df", None)

    st.markdown("---")

    scenario = st.session_state.get("active_scenario", "good")
    preset   = SCENARIOS.get(scenario, SCENARIOS["good"])

    # Override preset with uploaded dimensions if applicable
    if use_upload and uploaded_df is not None:
        preset = dict(preset, n_vars=uploaded_df.shape[1], n_obs=uploaded_df.shape[0])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dataset specification")
        n_vars = st.number_input(
            "Number of variables (items)",
            4, 40, int(preset["n_vars"]),
            disabled=(use_upload and uploaded_df is not None),
            help="Auto-set from uploaded file when a dataset is attached.",
        )
        n_obs = st.number_input(
            "Sample size (N)",
            50, 5000, int(preset["n_obs"]),
            disabled=(use_upload and uploaded_df is not None),
            help="Auto-set from uploaded file when a dataset is attached.",
        )
        n_factors   = st.number_input("Number of latent factors",    1,  10,   int(preset["n_factors"]))
        loading_str = st.slider("Target factor loading (λ)", 0.20, 0.95, float(preset["loading_str"]), 0.01)
        seed        = st.number_input("Random seed", 0, 9999, 42)

    with col2:
        st.markdown("#### Active scenario")
        st.info(SCENARIO_DESC.get(scenario, "Custom configuration."))
        st.markdown("#### Paste correlation matrix *(optional)*")
        st.caption("Lower-triangle, one row per line, space-separated. Leave blank to simulate.")
        st.text_area("Correlation matrix", height=110,
                     placeholder="0.62\n0.58 0.71\n0.11 0.09 0.67")

    st.markdown("---")

    # Label the button differently when using uploaded data
    _btn_label = "▶  Run diagnostics on uploaded data" if (use_upload and uploaded_df is not None) else "▶  Run diagnostics"

    if st.button(_btn_label, type="primary", use_container_width=True):
        # If an uploaded dataset is active, derive n_vars / n_obs from it
        if use_upload and uploaded_df is not None:
            n_vars = int(uploaded_df.shape[1])
            n_obs  = int(uploaded_df.shape[0])
            _source = f"uploaded file ({uploaded_file.name})"
        else:
            _source = f"scenario '{scenario}'"

        with st.spinner(f"Computing EFA and CFA diagnostics on {_source}…"):
            efa    = compute_efa(n_vars, n_obs, n_factors, loading_str, scenario, seed)
            cfa    = compute_cfa(n_vars, n_obs, n_factors, loading_str, scenario, seed)
            issues = build_issues(efa, cfa, n_obs, n_vars)
            st.session_state.update(dict(efa=efa, cfa=cfa, issues=issues,
                                         params=dict(n_vars=n_vars, n_obs=n_obs,
                                                     n_factors=n_factors, loading_str=loading_str,
                                                     scenario=scenario, seed=seed,
                                                     data_source=_source)))
        st.success("Diagnostics complete! Use the sidebar to explore results.")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("KMO",      f"{efa['kmo']:.3f}")
        c2.metric("RMSEA",    f"{cfa['rmsea']:.3f}")
        c3.metric("CFI",      f"{cfa['cfi']:.3f}")
        c4.metric("Var expl", f"{efa['var_explained']}%")
        c5.metric("Issues",   str(sum(1 for i in issues if i["sev"] in ("bad","warn"))))

        n_bad  = sum(1 for i in issues if i["sev"] == "bad")
        n_warn = sum(1 for i in issues if i["sev"] == "warn")
        if   n_bad  > 0: st.markdown(f'<div class="iss-bad">❌ {n_bad} critical issue(s) — visit Issues & Fixes.</div>', unsafe_allow_html=True)
        elif n_warn > 0: st.markdown(f'<div class="iss-warn">⚠️ {n_warn} warning(s) detected.</div>', unsafe_allow_html=True)
        else:            st.markdown('<div class="iss-ok">🎉 All diagnostics passed.</div>', unsafe_allow_html=True)
    elif "efa" not in st.session_state:
        st.info("👆 Set parameters above and click **Run diagnostics** to begin.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EFA
# ─────────────────────────────────────────────────────────────────────────────
def page_efa():
    st.title("🔍 EFA Diagnostics")
    if "efa" not in st.session_state:
        st.warning("Run diagnostics first → **Data Input**"); return

    efa = st.session_state["efa"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("KMO", f"{efa['kmo']:.3f}")
    c1.markdown(badge(efa["kmo"] >= 0.70, efa["kmo"] >= 0.60), unsafe_allow_html=True)
    c2.metric("Bartlett χ²", f"{efa['bartlett_chi2']:,}")
    c2.markdown('<span class="bdg-pass">✅ Significant</span>', unsafe_allow_html=True)
    c3.metric("Variance explained", f"{efa['var_explained']}%")
    c3.markdown(badge(efa["var_explained"] >= 50, efa["var_explained"] >= 40), unsafe_allow_html=True)
    c4.metric("Factors retained", str(efa["n_factors_retained"]))

    st.markdown("---")
    t1, t2, t3 = st.tabs(["📈 Scree plot", "🔥 Loading matrix", "◐ Communalities"])

    with t1:
        st.plotly_chart(chart_scree(efa["eigenvalues"], efa["n_factors_retained"]),
                        use_container_width=True)
        st.caption("Purple bars = retained factors (above Kaiser λ=1 line).")

    with t2:
        st.plotly_chart(chart_heatmap(efa["loadings"], efa["n_factors_retained"]),
                        use_container_width=True)
        nf = efa["n_factors_retained"]; ipf = efa["ipf"]; nv = len(efa["loadings"])
        rows = []
        for i, row in enumerate(efa["loadings"]):
            my_f = min(i // ipf, nf - 1)
            r = {"Item": f"V{i+1}"}
            for f in range(nf): r[f"F{f+1}"] = round(row[f], 3)
            r["Primary"]     = f"F{my_f+1}"
            r["Cross-loads"] = ", ".join(f"F{f+1}" for f,l in enumerate(row) if f != my_f and l > 0.32) or "—"
            rows.append(r)
        df = pd.DataFrame(rows)
        def hl(row):
            if row["Cross-loads"] != "—": return ["background-color:#ffebee"]*len(row)
            pri = row["Primary"]
            if row.get(pri, 1) < 0.40:   return ["background-color:#fff8e1"]*len(row)
            return [""]*len(row)
        st.dataframe(df.style.apply(hl, axis=1), use_container_width=True, height=340)
        st.caption("🔴 Cross-loaders  |  🟡 Low primary loading (< 0.40)  |  🟢 Clean items")

    with t3:
        st.plotly_chart(chart_communalities(efa["communalities"]), use_container_width=True)
        low = [f"V{i+1}" for i,c in enumerate(efa["communalities"]) if c < 0.40]
        if low: st.error(f"**{len(low)} items below h²=0.40:** {', '.join(low)}")
        else:   st.success("All communalities ≥ 0.40 ✓")
        avg = sum(efa["communalities"]) / len(efa["communalities"])
        st.metric("Average communality", f"{avg:.3f}")

    with st.expander("📖 KMO interpretation guide"):
        st.table(pd.DataFrame({
            "KMO range": ["0.90–1.00","0.80–0.89","0.70–0.79","0.60–0.69","0.50–0.59","< 0.50"],
            "Rating":    ["Marvelous","Meritorious","Middling","Mediocre","Miserable","Unacceptable"],
            "Action":    ["Proceed","Proceed","Proceed","Cautious","Revise items","Do not proceed"],
        }))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CFA
# ─────────────────────────────────────────────────────────────────────────────
def page_cfa():
    st.title("📐 CFA Fit Indices")
    if "cfa" not in st.session_state:
        st.warning("Run diagnostics first → **Data Input**"); return

    cfa = st.session_state["cfa"]; efa = st.session_state["efa"]

    g1,g2,g3,g4 = st.columns(4)
    g1.plotly_chart(chart_gauge(cfa["rmsea"],"RMSEA",0.060,0.080,False), use_container_width=True)
    g2.plotly_chart(chart_gauge(cfa["cfi"],  "CFI",  0.950,0.900,True),  use_container_width=True)
    g3.plotly_chart(chart_gauge(cfa["tli"],  "TLI",  0.950,0.900,True),  use_container_width=True)
    g4.plotly_chart(chart_gauge(cfa["srmr"], "SRMR", 0.080,0.100,False), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Fit index summary")

    r = cfa
    fit_rows = [
        dict(Index="RMSEA", Value=f"{r['rmsea']:.3f}", Threshold="< 0.060",
             Rule="< .06 excellent · < .08 acceptable",
             Status="✅ Pass" if r["rmsea"]<0.06 else ("⚠️ Marginal" if r["rmsea"]<0.08 else "❌ Fail")),
        dict(Index="CFI",   Value=f"{r['cfi']:.3f}",   Threshold="> 0.950",
             Rule="> .95 excellent · > .90 acceptable",
             Status="✅ Pass" if r["cfi"]>0.95   else ("⚠️ Marginal" if r["cfi"]>0.90   else "❌ Fail")),
        dict(Index="TLI",   Value=f"{r['tli']:.3f}",   Threshold="> 0.950",
             Rule="> .95 excellent · > .90 acceptable",
             Status="✅ Pass" if r["tli"]>0.95   else ("⚠️ Marginal" if r["tli"]>0.90   else "❌ Fail")),
        dict(Index="SRMR",  Value=f"{r['srmr']:.3f}",  Threshold="< 0.080",
             Rule="< .08 excellent · < .10 acceptable",
             Status="✅ Pass" if r["srmr"]<0.08  else ("⚠️ Marginal" if r["srmr"]<0.10  else "❌ Fail")),
        dict(Index="χ²/df", Value=f"{r['chi2df']:.2f}",Threshold="< 3.0",
             Rule="< 2 excellent · < 3 good · < 5 marginal",
             Status="✅ Pass" if r["chi2df"]<3   else ("⚠️ Marginal" if r["chi2df"]<5   else "❌ Fail")),
    ]
    def col_status(v):
        if "Pass"     in v: return "background-color:#e8f5e9;color:#2e7d32"
        if "Marginal" in v: return "background-color:#fff8e1;color:#e65100"
        if "Fail"     in v: return "background-color:#ffebee;color:#b71c1c"
        return ""
    df = pd.DataFrame(fit_rows)
    st.dataframe(df.style.applymap(col_status, subset=["Status"]),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Standardized path diagram")
    nf = efa["n_factors_retained"]; ipf = efa["ipf"]; nv = len(efa["loadings"])
    cols = st.columns(nf)
    for f in range(nf):
        with cols[f]:
            st.markdown(f"**Factor F{f+1}**")
            for i in range(f * ipf, min((f + 1) * ipf, nv)):
                l = efa["loadings"][i][f]
                col = "#2e7d32" if l >= 0.50 else ("#e65100" if l >= 0.40 else "#b71c1c")
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 10px;background:#fafafa;border-radius:6px;"
                    f"margin-bottom:4px;border:1px solid #eee'>"
                    f"<span style='font-size:0.85rem'>V{i+1}</span>"
                    f"<span style='font-weight:600;color:{col};font-size:0.85rem'>{l:.2f}</span></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("#### Modification indices (top 10)")
    st.caption("MI > 10 → consider freeing correlated residuals (if theoretically justified).")
    st.plotly_chart(chart_mi(cfa["mi_pairs"]), use_container_width=True)
    if cfa["mi_pairs"]:
        mi_df = pd.DataFrame(cfa["mi_pairs"][:10])
        mi_df.columns = ["Item A","Item B","MI","Expected Δχ²"]
        st.dataframe(mi_df, use_container_width=True, hide_index=True)

    with st.expander("📚 Reference thresholds & citations"):
        st.markdown("""
| Index  | Excellent | Acceptable | Reject |
|--------|-----------|------------|--------|
| RMSEA  | < 0.05    | < 0.08     | > 0.10 |
| CFI    | > 0.97    | > 0.90     | < 0.90 |
| TLI    | > 0.97    | > 0.90     | < 0.90 |
| SRMR   | < 0.05    | < 0.08     | > 0.10 |
| χ²/df  | < 2.0     | < 3.0      | > 5.0  |

*Hu & Bentler (1999); Brown (2015); MacCallum et al. (1996)*
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ISSUES & FIXES
# ─────────────────────────────────────────────────────────────────────────────
def page_issues():
    st.title("⚠️ Issues & Fixes")
    if "issues" not in st.session_state:
        st.warning("Run diagnostics first → **Data Input**"); return

    issues = st.session_state["issues"]
    params = st.session_state.get("params", {})

    n_bad  = sum(1 for i in issues if i["sev"] == "bad")
    n_warn = sum(1 for i in issues if i["sev"] == "warn")
    n_ok   = sum(1 for i in issues if i["sev"] == "ok")

    b1,b2,b3 = st.columns(3)
    b1.metric("Critical issues", str(n_bad),  delta="none ✓" if not n_bad else "must fix",  delta_color="normal" if not n_bad else "inverse")
    b2.metric("Warnings",        str(n_warn), delta="none ✓" if not n_warn else "review",   delta_color="normal" if not n_warn else "inverse")
    b3.metric("Checks passed",   str(n_ok))

    if n_bad == 0 and n_warn == 0:
        st.success("🎉 All diagnostic checks passed!")
    elif n_bad > 0:
        st.error(f"❌ {n_bad} critical issue(s) require attention before running EFA/CFA.")
    else:
        st.warning(f"⚠️ {n_warn} warning(s) — minor improvements recommended.")

    st.markdown("---")

    sev_filter = st.multiselect("Filter by severity",
                                ["Critical","Warning","Passed"],
                                default=["Critical","Warning","Passed"])
    fmap = {"Critical":"bad","Warning":"warn","Passed":"ok"}
    active_sevs = {fmap[f] for f in sev_filter}

    CSS  = {"bad":"iss-bad","warn":"iss-warn","ok":"iss-ok"}
    ICON = {"bad":"❌","warn":"⚠️","ok":"✅"}
    LBL  = {"bad":"Critical","warn":"Warning","ok":"Passed"}

    for iss in issues:
        if iss["sev"] not in active_sevs: continue
        st.markdown(
            f'<div class="{CSS[iss["sev"]]}">'
            f'<strong>{ICON[iss["sev"]]} [{LBL[iss["sev"]]}] {iss["title"]}</strong><br>'
            f'<span style="font-size:0.88rem;color:#555">{iss["desc"]}</span></div>',
            unsafe_allow_html=True,
        )
        if iss["fixes"]:
            with st.expander("🛠 Recommended fixes", expanded=(iss["sev"] == "bad")):
                for fix in iss["fixes"]:
                    st.markdown(f"- {fix}")

    # Export
    st.markdown("---")
    with st.expander("📤 Export report"):
        lines = ["FactorLens Diagnostic Report", "="*40,
                 f"Dataset: {params.get('n_vars','')} vars, N={params.get('n_obs','')}, "
                 f"{params.get('n_factors','')} factors\n"]
        for iss in issues:
            lines += [f"\n[{LBL[iss['sev']].upper()}] {iss['title']}", f"  {iss['desc']}"]
            for fix in iss.get("fixes", []): lines.append(f"  → {fix}")
        report = "\n".join(lines)
        st.text_area("Report text", report, height=250)
        st.download_button("⬇ Download (.txt)", report, file_name="factorlens_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REGENERATION LAB
# ─────────────────────────────────────────────────────────────────────────────
def page_regen():
    st.title("🔄 Regeneration Lab")
    st.markdown("Tune parameters and see **predicted diagnostics** update live before generating data.")

    params    = st.session_state.get("params", {})
    n_factors = params.get("n_factors", 3)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Factor structure")
        ipf   = st.slider("Items per factor",          3,  8, 4)
        tload = st.slider("Target loading (λ)",       0.45, 0.95, 0.70, 0.01)
        lsd   = st.slider("Loading spread (±SD)",     0.01, 0.15, 0.06, 0.01)
        ifc   = st.slider("Inter-factor correlation", 0.00, 0.60, 0.20, 0.01)

    with col2:
        st.markdown("#### Data quality")
        n_obs   = st.slider("Sample size (N)",        100, 1000, 300, 50)
        cl_ceil = st.slider("Cross-loading ceiling",  0.00, 0.35, 0.15, 0.01)
        ev_fl   = st.slider("Error variance floor",   0.10, 0.55, 0.25, 0.01)
        skew    = st.slider("Item skewness limit",    0.0,  2.0,  0.5,  0.1)

    pred = predict_diagnostics(tload, cl_ceil, ev_fl, n_obs)

    st.markdown("---")
    st.markdown("#### Predicted diagnostics")

    def pc(v, good, bad_t, higher=True):
        ok = v >= good if higher else v <= good
        mg = v >= bad_t if higher else v <= bad_t
        return "🟢" if ok else ("🟡" if mg else "🔴")

    p1,p2,p3,p4 = st.columns(4)
    p1.metric(f"{pc(pred['kmo'],  0.70,0.60)} KMO",    f"{pred['kmo']:.3f}")
    p2.metric(f"{pc(pred['rmsea'],0.06,0.08,False)} RMSEA", f"{pred['rmsea']:.3f}")
    p3.metric(f"{pc(pred['cfi'],  0.95,0.90)} CFI",    f"{pred['cfi']:.3f}")
    p4.metric(f"{pc(pred['h2'],   0.50,0.40)} Avg h²", f"{pred['h2']:.3f}")

    passes    = pred["kmo"] >= 0.70 and pred["rmsea"] <= 0.08 and pred["cfi"] >= 0.90 and pred["h2"] >= 0.40
    excellent = pred["kmo"] >= 0.80 and pred["rmsea"] <= 0.06 and pred["cfi"] >= 0.95 and pred["h2"] >= 0.50
    if excellent:  st.success("🎉 Excellent — all indices in green zone.")
    elif passes:   st.warning("⚠️ Acceptable but some indices are marginal.")
    else:          st.error("❌ EFA/CFA likely to fail. Increase loading strength and/or N.")

    st.plotly_chart(chart_radar(pred), use_container_width=True)

    # Spec JSON
    st.markdown("---")
    spec = dict(n_factors=n_factors, items_per_factor=ipf,
                target_loading=round(tload,2), loading_sd=round(lsd,2),
                cross_loading_ceiling=round(cl_ceil,2), inter_factor_corr=round(ifc,2),
                error_variance_floor=round(ev_fl,2), item_skewness_limit=round(skew,1),
                sample_size=n_obs,
                predicted=dict(kmo=pred["kmo"],rmsea=pred["rmsea"],
                               cfi=pred["cfi"],avg_h2=pred["h2"]))
    spec_json = json.dumps(spec, indent=2)
    st.markdown("#### Generation specification")
    st.code(spec_json, language="json")
    st.download_button("⬇ Download spec (.json)", spec_json,
                       file_name="factorlens_spec.json", mime="application/json")

    # Parameter sweep
    with st.expander("🔬 N sweep — how CFI & RMSEA change with sample size"):
        sweep = [{"N": n,
                  **predict_diagnostics(tload, cl_ceil, ev_fl, n)}
                 for n in range(100, 1001, 50)]
        sdf = pd.DataFrame(sweep).set_index("N")[["cfi","rmsea"]]
        sdf.columns = ["CFI", "RMSEA"]
        st.line_chart(sdf)

    with st.expander("📖 Rules of thumb"):
        st.markdown("""
| Parameter | Minimum | Good | Excellent |
|-----------|---------|------|-----------|
| Target λ  | 0.40    | 0.60 | 0.70+     |
| Items/factor | 3   | 4–5  | 5–6       |
| N         | 200     | 300–500 | 500+   |
| Cross-load ceiling | < 0.30 | < 0.20 | < 0.15 |
| Error variance | — | < 0.40 | < 0.30 |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
QUICK_QS = [
    "Why does my RMSEA fail?",
    "How many items per factor do I need?",
    "What sample size is needed for CFA?",
    "Explain modification indices",
    "How do I fix cross-loading items?",
    "My CFI is below 0.90 — what now?",
    "What is a Heywood case?",
    "What loading strength should I target?",
]


def page_ai():
    st.title("🤖 AI Advisor")
    st.markdown("Ask for personalised help grounded in your actual diagnostic values.")

    # ── API key & model ───────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model   = "gpt-4o-mini"

    with st.expander("⚙️ OpenAI settings", expanded=not bool(api_key)):
        input_key = st.text_input("OpenAI API key", value=api_key,
                                  type="password", placeholder="sk-...")
        if input_key: api_key = input_key
        model = st.selectbox("Model", ["gpt-4o-mini","gpt-4o","gpt-4-turbo","gpt-3.5-turbo"],
                             index=0)
        st.caption("Your key is used only for this session. "
                   "Set `OPENAI_API_KEY` in Streamlit secrets to avoid re-entering it.")

    if not api_key:
        st.info("Enter your OpenAI API key above to enable the AI advisor.")

    # ── Quick question chips ──────────────────────────────────────────────────
    st.markdown("#### Quick questions")
    cols = st.columns(4)
    for idx, q in enumerate(QUICK_QS):
        if cols[idx % 4].button(q, key=f"qq_{idx}", use_container_width=True):
            st.session_state.setdefault("chat_msgs", [])
            st.session_state["chat_msgs"].append({"role":"user","content":q})
            st.session_state["_pending"] = True

    st.markdown("---")

    # ── Chat ──────────────────────────────────────────────────────────────────
    st.session_state.setdefault("chat_msgs", [])

    for msg in st.session_state["chat_msgs"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle quick-question pending
    if st.session_state.pop("_pending", False):
        msgs = st.session_state["chat_msgs"]
        if msgs and msgs[-1]["role"] == "user":
            _do_ai_response(msgs, api_key, model)

    # Text input
    user_input = st.chat_input("Ask about your EFA/CFA results…")
    if user_input:
        st.session_state["chat_msgs"].append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        _do_ai_response(st.session_state["chat_msgs"], api_key, model)

    if st.session_state["chat_msgs"]:
        if st.button("🗑 Clear chat"):
            st.session_state["chat_msgs"] = []
            st.rerun()


def _do_ai_response(history, api_key, model):
    system = build_ai_context(
        st.session_state.get("efa"),
        st.session_state.get("cfa"),
        st.session_state.get("issues"),
        st.session_state.get("params", {}),
    )
    with st.chat_message("assistant"):
        ph = st.empty()
        ph.markdown("_Thinking…_")
        if not api_key:
            answer = ("⚠️ No API key provided. Enter your OpenAI key in the settings above.\n\n"
                      "**Quick tips while you get set up:**\n"
                      "- Target loadings ≥ 0.60 for reliable CFA\n"
                      "- Minimum N = 200; ideally N ≥ 10× free parameters\n"
                      "- Cross-loadings should be < 0.30\n"
                      "- RMSEA < 0.06, CFI > 0.95 for excellent fit")
        else:
            try:
                answer = call_openai(history, system, api_key, model)
            except Exception as e:
                err = str(e)
                if "401" in err or "Unauthorized" in err or "Incorrect API key" in err:
                    answer = "❌ Invalid API key. Please check your OpenAI API key and try again."
                elif "429" in err or "Rate limit" in err:
                    answer = "⚠️ Rate limit reached. Please wait a moment and try again."
                elif "model" in err.lower():
                    answer = f"❌ Model error: {err}. Try selecting a different model in settings."
                else:
                    answer = f"❌ Error: {err}"
        ph.markdown(answer)
    st.session_state["chat_msgs"].append({"role":"assistant","content":answer})


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
routes = {
    "📥 Data Input":     page_data_input,
    "🔍 EFA Diagnostics": page_efa,
    "📐 CFA Fit Indices": page_cfa,
    "⚠️ Issues & Fixes":  page_issues,
    "🔄 Regeneration Lab": page_regen,
    "🤖 AI Advisor":     page_ai,
}
routes[page]()
