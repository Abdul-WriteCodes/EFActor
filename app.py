"""
╔══════════════════════════════════════════════════════════════════╗
║         EFA / CFA ANALYSER  —  Single-File Streamlit App        ║
║  Upload → Suitability → EFA → Diagnose → Fix → CFA → Export     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io
import zipfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

# ── sklearn >= 1.6 compatibility ─────────────────────────────────────────────
# factor_analyzer 0.5.x uses force_all_finite, renamed ensure_all_finite in sklearn 1.6
try:
    import factor_analyzer.factor_analyzer as _fa_mod
    import factor_analyzer.confirmatory_factor_analyzer as _cfa_mod
    from sklearn.utils.validation import check_array as _orig_check_array
    def _compat_check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            val = kwargs.pop("force_all_finite")
            kwargs["ensure_all_finite"] = (val is True)
        return _orig_check_array(*args, **kwargs)
    _fa_mod.check_array  = _compat_check_array
    _cfa_mod.check_array = _compat_check_array
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EFA / CFA Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════
C = dict(
    accent="#6c8dfa", accent2="#a78bfa",
    green="#34d399", red="#f87171", yellow="#fbbf24",
    bg="#0f1117", surface="#1a1d27", border="#2d3148",
    text="#e2e8f0", muted="#94a3b8",
)
LAYOUT_BASE = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Georgia, serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[C["accent"], C["accent2"], C["green"], C["yellow"], C["red"]],
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --accent:#6c8dfa; --accent2:#a78bfa; --green:#34d399;
  --red:#f87171; --yellow:#fbbf24; --border:#2d3148;
  --surface:#1a1d27; --muted:#94a3b8;
}
html,body,[class*="css"]{font-family:'Crimson Pro',Georgia,serif;}
.stApp{background:#0f1117;}
[data-testid="stSidebar"]{background:#13161f !important;border-right:1px solid var(--border);}
h1{color:var(--accent) !important;letter-spacing:-0.5px;}
h2{color:var(--accent2) !important;border-bottom:1px solid var(--border);padding-bottom:8px;}
h3{color:#e2e8f0 !important;}
.step-badge{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#1a1d27,#2d3148);
  border:1px solid var(--border);border-left:3px solid var(--accent);padding:10px 18px;border-radius:6px;
  color:#e2e8f0;font-size:1.05rem;margin-bottom:16px;width:100%;}
.step-num{background:var(--accent);color:#0f1117;border-radius:50%;width:26px;height:26px;
  display:inline-flex;align-items:center;justify-content:center;font-size:.85rem;font-weight:bold;flex-shrink:0;}
.pill-pass{background:rgba(52,211,153,.15);color:var(--green);border:1px solid var(--green);
  border-radius:20px;padding:2px 12px;font-size:.85rem;font-weight:600;display:inline-block;}
.pill-fail{background:rgba(248,113,113,.15);color:var(--red);border:1px solid var(--red);
  border-radius:20px;padding:2px 12px;font-size:.85rem;font-weight:600;display:inline-block;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 20px;text-align:center;}
.metric-val{font-size:2rem;font-weight:600;color:var(--accent);}
.metric-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:4px;}
.info-box{background:rgba(108,141,250,.08);border:1px solid rgba(108,141,250,.25);border-radius:6px;
  padding:12px 16px;color:#c7d2fe;font-size:.92rem;margin:8px 0;}
.warn-box{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);border-radius:6px;
  padding:12px 16px;color:#fde68a;font-size:.92rem;margin:8px 0;}
.section-divider{border:none;border-top:1px solid var(--border);margin:28px 0;}
code,pre{font-family:'JetBrains Mono',monospace !important;}
.stButton>button{background:linear-gradient(135deg,#1d2035,#2d3560) !important;
  border:1px solid var(--accent) !important;color:var(--accent) !important;
  font-family:'Crimson Pro',serif !important;font-size:1rem !important;border-radius:6px !important;}
.stButton>button:hover{background:linear-gradient(135deg,var(--accent),#4f6fe8) !important;
  color:#0f1117 !important;border-color:transparent !important;}
.stDownloadButton>button{background:linear-gradient(135deg,#1a2d1a,#1f3b1f) !important;
  border:1px solid var(--green) !important;color:var(--green) !important;border-radius:6px !important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 1: EFA FUNCTIONS ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def check_efa_suitability(df: pd.DataFrame) -> dict:
    kmo_all, kmo_model = calculate_kmo(df)
    chi2, p = calculate_bartlett_sphericity(df)
    labels = {.9:"Marvellous",.8:"Meritorious",.7:"Middling",.6:"Mediocre",.5:"Miserable"}
    kmo_label = next((v for k, v in sorted(labels.items(), reverse=True) if kmo_model >= k), "Unacceptable")
    return dict(
        kmo_model=round(float(kmo_model), 4), kmo_label=kmo_label,
        kmo_pass=kmo_model >= 0.6,
        bartlett_chi2=round(float(chi2), 4), bartlett_p=round(float(p), 6),
        bartlett_pass=p < 0.05,
        overall_pass=kmo_model >= 0.6 and p < 0.05,
    )


def determine_n_factors(df: pd.DataFrame) -> dict:
    max_factors = max(1, min(len(df.columns) - 1, len(df) - 1))
    fa = FactorAnalyzer(n_factors=max_factors, rotation=None)
    fa.fit(df)
    ev, _ = fa.get_eigenvalues()
    return dict(eigenvalues=ev.tolist(), suggested_n=max(1, int(np.sum(ev > 1))))


def run_efa(df: pd.DataFrame, n_factors: int, rotation: str = "varimax") -> dict:
    n_factors = min(n_factors, len(df.columns) - 1)
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)
    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=factor_labels)
    communalities = pd.Series(fa.get_communalities(), index=df.columns, name="Communality")
    variance = pd.DataFrame(
        fa.get_factor_variance(),
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_labels,
    ).T
    return dict(loadings=loadings, communalities=communalities,
                variance=variance, n_factors=n_factors)


def diagnose_loadings(loadings: pd.DataFrame, communalities: pd.Series,
                      load_thresh: float = 0.4, comm_thresh: float = 0.3) -> pd.DataFrame:
    records = []
    for var in loadings.index:
        abs_row = np.abs(loadings.loc[var])
        max_load = abs_row.max()
        n_high = int((abs_row >= load_thresh).sum())
        comm = float(communalities[var])
        issues, severity = [], 0.0
        if comm < comm_thresh:
            issues.append("Low Communality"); severity += (comm_thresh - comm) * 3
        if n_high == 0:
            issues.append("Weak Loader"); severity += (load_thresh - max_load) * 2
        elif n_high > 1:
            issues.append("Cross-Loader")
            s = sorted(abs_row.values, reverse=True)
            severity += (1 - (s[0] - s[1])) * 2
        records.append(dict(
            Variable=var, MaxLoading=round(max_load, 4),
            FactorsAboveThreshold=n_high, Communality=round(comm, 4),
            Issue=", ".join(issues) if issues else "OK",
            Severity=round(severity, 4), RecommendDrop=len(issues) > 0,
        ))
    return pd.DataFrame(records).sort_values("Severity", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# ── SECTION 2: CFA FUNCTIONS ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def build_cfa_model(loadings: pd.DataFrame, threshold: float = 0.4) -> tuple:
    factor_vars = {}
    for var in loadings.index:
        abs_row = np.abs(loadings.loc[var])
        if abs_row.max() >= threshold:
            best = abs_row.idxmax()
            factor_vars.setdefault(best, []).append(var)
    factor_vars = {f: v for f, v in factor_vars.items() if len(v) >= 2}
    model_str = "\n".join(f"{f} =~ " + " + ".join(v) for f, v in factor_vars.items())
    return model_str, factor_vars


def run_cfa(df: pd.DataFrame, model_str: str) -> dict:
    try:
        from semopy import Model
        from semopy.stats import calc_stats
        model = Model(model_str)
        model.fit(df)
        try:
            raw_stats = calc_stats(model)
            fit_indices = _parse_fit_indices(raw_stats)
        except Exception as e:
            fit_indices = {"parse_error": str(e)}
        return dict(success=True, fit_indices=fit_indices,
                    estimates=model.inspect(), model_str=model_str, error=None)
    except Exception as e:
        return dict(success=False, fit_indices={}, estimates=None,
                    model_str=model_str, error=str(e))


def _parse_fit_indices(stats) -> dict:
    flat = stats.iloc[0] if isinstance(stats, pd.DataFrame) else stats
    flat.index = [str(c).strip().upper() for c in flat.index]
    mapping = dict(
        CFI=["CFI"], TLI=["TLI","NNFI"], RMSEA=["RMSEA"], SRMR=["SRMR"],
        Chi2=["CHI2","CHISQ","CHI-SQUARE","X2"], df=["DF","DOF"],
        p_value=["P-VALUE","PVALUE","P_VALUE","P(CHI2)"], AIC=["AIC"], BIC=["BIC"],
    )
    result = {}
    for label, candidates in mapping.items():
        for c in candidates:
            if c in flat.index:
                try: result[label] = round(float(flat[c]), 4)
                except: result[label] = flat[c]
                break
    return result


def assess_cfa_fit(fit_indices: dict, thresholds: dict) -> dict:
    assessment = {}
    for idx, direction in [("CFI","≥"),("TLI","≥"),("RMSEA","≤"),("SRMR","≤")]:
        if idx in fit_indices and idx in thresholds:
            val, thresh = fit_indices[idx], thresholds[idx]
            passed = val >= thresh if direction == "≥" else val <= thresh
            assessment[idx] = dict(value=val, threshold=thresh, pass_=passed, direction=direction)
    n_pass = sum(1 for v in assessment.values() if v["pass_"])
    return dict(indices=assessment, n_pass=n_pass,
                n_total=len(assessment),
                overall_pass=(len(assessment) > 0 and n_pass == len(assessment)))


def get_modification_suggestions(fit_assessment: dict) -> list:
    suggestions, indices = [], fit_assessment.get("indices", {})
    checks = [
        ("RMSEA", "RMSEA exceeds threshold. Consider freeing residual covariances between items sharing method variance, or removing items with high modification indices."),
        ("CFI",   "CFI is below threshold. Check whether indicators load on multiple factors. Consider adding cross-loadings or removing weak indicators."),
        ("SRMR",  "SRMR exceeds threshold. Large residual correlations exist — review for systematic patterns among residuals."),
    ]
    for idx, msg in checks:
        if idx in indices and not indices[idx]["pass_"]:
            suggestions.append(f"{idx} = {indices[idx]['value']:.3f} — {msg}")
    if not suggestions and not fit_assessment.get("overall_pass"):
        suggestions.append("Model fit is inadequate. Consider revising factor structure, removing low-communality items, or allowing correlated residuals for items with common wording.")
    return suggestions


# ══════════════════════════════════════════════════════════════════
# ── SECTION 3: SYNTHETIC DATA FUNCTIONS ──────────────────────────
# ══════════════════════════════════════════════════════════════════

def _make_psd(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ev, evec = np.linalg.eigh(matrix)
    return evec @ np.diag(np.maximum(ev, eps)) @ evec.T


def generate_factor_based(df: pd.DataFrame, efa_result: dict,
                           n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    loadings = efa_result["loadings"].values
    communalities = efa_result["communalities"].values
    n_factors, columns = efa_result["n_factors"], efa_result["loadings"].index.tolist()
    factor_scores = np.random.multivariate_normal(np.zeros(n_factors), np.eye(n_factors), n_samples)
    common = factor_scores @ loadings.T
    unique = np.random.normal(0, np.sqrt(np.maximum(1 - communalities, 1e-6)), (n_samples, len(columns)))
    synthetic = common + unique
    orig_mean, orig_std = df[columns].mean().values, df[columns].std().values
    orig_std = np.where(orig_std == 0, 1.0, orig_std)
    syn_mean, syn_std = synthetic.mean(0), synthetic.std(0)
    syn_std = np.where(syn_std == 0, 1.0, syn_std)
    rescaled = ((synthetic - syn_mean) / syn_std) * orig_std + orig_mean
    return pd.DataFrame(rescaled, columns=columns)


def generate_correlation_based(df: pd.DataFrame, n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    cov = _make_psd(df.cov().values)
    synthetic = np.random.multivariate_normal(df.mean().values, cov, n_samples)
    return pd.DataFrame(synthetic, columns=df.columns.tolist())


def validate_synthetic(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(
        Variable=col,
        OrigMean=round(original[col].mean(), 4), SynMean=round(synthetic[col].mean(), 4),
        OrigStd=round(original[col].std(), 4),  SynStd=round(synthetic[col].std(), 4),
        MeanDelta=round(abs(original[col].mean() - synthetic[col].mean()), 4),
        StdDelta=round(abs(original[col].std() - synthetic[col].std()), 4),
    ) for col in original.columns])


# ══════════════════════════════════════════════════════════════════
# ── SECTION 4: PLOT FUNCTIONS ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

def plot_scree(eigenvalues: list, suggested_n: int) -> go.Figure:
    x = list(range(1, len(eigenvalues) + 1))
    colors = [C["green"] if i < suggested_n else C["muted"] for i in range(len(eigenvalues))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=eigenvalues, marker_color=colors, name="Eigenvalue",
                         hovertemplate="Factor %{x}<br>λ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=eigenvalues, mode="lines+markers",
                             line=dict(color=C["accent"], width=2),
                             marker=dict(size=7, color=C["accent"]), showlegend=False))
    fig.add_hline(y=1.0, line_dash="dash", line_color=C["yellow"],
                  annotation_text="Kaiser criterion (λ=1)", annotation_font_color=C["yellow"])
    fig.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                      title=dict(text=f"Scree Plot — Suggested factors: <b>{suggested_n}</b>",
                                 font=dict(color=C["accent2"])),
                      xaxis=dict(title="Factor", gridcolor=C["border"], tickmode="linear"),
                      yaxis=dict(title="Eigenvalue", gridcolor=C["border"]))
    return fig


def plot_loading_heatmap(loadings: pd.DataFrame, threshold: float = 0.4) -> go.Figure:
    z = loadings.values
    variables, factors = loadings.index.tolist(), loadings.columns.tolist()
    annotations = [dict(x=f, y=v, text=f"{z[i][j]:.2f}", showarrow=False,
                        font=dict(color="white" if abs(z[i][j]) >= threshold else C["muted"], size=11))
                   for i, v in enumerate(variables) for j, f in enumerate(factors)]
    fig = go.Figure(go.Heatmap(
        z=z, x=factors, y=variables, zmid=0, zmin=-1, zmax=1,
        colorscale=[[0,"#1e1b4b"],[0.5,C["surface"]],[1,C["accent"]]],
        colorbar=dict(title="Loading", tickfont=dict(color=C["text"]), title_font=dict(color=C["text"])),
        hovertemplate="%{y} → %{x}<br>Loading: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, annotations=annotations,
                      title=dict(text="Factor Loading Heatmap", font=dict(color=C["accent2"])),
                      height=max(300, len(variables) * 32 + 100), xaxis=dict(side="top"))
    return fig


def plot_communalities(communalities: pd.Series, comm_thresh: float = 0.3) -> go.Figure:
    colors = [C["green"] if v >= 0.5 else (C["yellow"] if v >= comm_thresh else C["red"])
              for v in communalities.values]
    fig = go.Figure(go.Bar(x=communalities.index.tolist(), y=communalities.values,
                           marker_color=colors,
                           hovertemplate="%{x}<br>Communality: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=comm_thresh, line_dash="dash", line_color=C["red"],
                  annotation_text=f"Threshold ({comm_thresh})", annotation_font_color=C["red"])
    fig.add_hline(y=0.5, line_dash="dot", line_color=C["yellow"],
                  annotation_text="Good (0.50)", annotation_font_color=C["yellow"])
    fig.update_layout(**LAYOUT_BASE, height=360,
                      title=dict(text="Communalities per Variable", font=dict(color=C["accent2"])),
                      xaxis=dict(title="Variable", gridcolor=C["border"], tickangle=-35),
                      yaxis=dict(title="Communality", gridcolor=C["border"], range=[0, 1]))
    return fig


def plot_fit_indices(fit_assessment: dict) -> go.Figure:
    indices = fit_assessment.get("indices", {})
    if not indices:
        return go.Figure()
    labels = list(indices.keys())
    values = [d["value"] for d in indices.values()]
    colors = [C["green"] if d["pass_"] else C["red"] for d in indices.values()]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"{v:.3f}" for v in values], textposition="outside",
                           hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    for i, (label, data) in enumerate(indices.items()):
        fig.add_annotation(x=label, y=data["threshold"],
                           text=f"Threshold: {data['threshold']}",
                           showarrow=True, arrowhead=2, arrowcolor=C["yellow"],
                           font=dict(color=C["yellow"], size=10), ay=-30)
    fig.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                      title=dict(text="CFA Fit Indices", font=dict(color=C["accent2"])),
                      xaxis=dict(title="Index", gridcolor=C["border"]),
                      yaxis=dict(title="Value", gridcolor=C["border"]))
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True).round(2)
    cols = corr.columns.tolist()
    annotations = [dict(x=c, y=r, text=f"{corr.loc[r,c]:.2f}", showarrow=False,
                        font=dict(size=9, color="white" if abs(corr.loc[r,c]) > 0.5 else C["text"]))
                   for r in corr.index for c in cols]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="r", tickfont=dict(color=C["text"]), title_font=dict(color=C["text"])),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, annotations=annotations,
                      title=dict(text="Correlation Matrix", font=dict(color=C["accent2"])),
                      height=max(350, len(cols) * 30 + 120), xaxis=dict(tickangle=-35))
    return fig


def plot_synthetic_comparison(original: pd.DataFrame, synthetic: pd.DataFrame,
                               max_vars: int = 6) -> go.Figure:
    cols = original.columns[:max_vars].tolist()
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
    for i, col in enumerate(cols):
        r, ci = i // ncols + 1, i % ncols + 1
        fig.add_trace(go.Histogram(x=original[col], name="Original", nbinsx=20,
                                   marker_color=C["accent"], opacity=0.6, showlegend=(i==0)), row=r, col=ci)
        fig.add_trace(go.Histogram(x=synthetic[col], name="Synthetic", nbinsx=20,
                                   marker_color=C["accent2"], opacity=0.6, showlegend=(i==0)), row=r, col=ci)
    fig.update_layout(**LAYOUT_BASE, barmode="overlay", height=300 * nrows,
                      title=dict(text="Original vs Synthetic Distributions", font=dict(color=C["accent2"])),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center",
                                  font=dict(color=C["text"])))
    return fig


# ══════════════════════════════════════════════════════════════════
# ── SECTION 5: HTML REPORT GENERATOR ─────────────────────────────
# ══════════════════════════════════════════════════════════════════

def generate_html_report(original_df, cleaned_df, suitability, efa_result,
                          diagnostics, dropped_vars, cfa_result, fit_assessment,
                          cfa_thresholds, synthetic_validation=None, model_str="") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EFA/CFA Report</title>
<style>
:root{{--bg:#0f1117;--surface:#1a1d27;--border:#2d3148;--accent:#6c8dfa;
  --accent2:#a78bfa;--green:#34d399;--red:#f87171;--yellow:#fbbf24;
  --text:#e2e8f0;--muted:#94a3b8;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:Georgia,serif;line-height:1.7;padding:40px 20px;}}
.container{{max-width:960px;margin:0 auto;}}
h1{{font-size:2.2rem;color:var(--accent);margin-bottom:4px;}}
h2{{font-size:1.2rem;color:var(--accent2);margin:32px 0 12px;border-bottom:1px solid var(--border);
  padding-bottom:6px;text-transform:uppercase;letter-spacing:1px;}}
.meta{{color:var(--muted);font-size:.9rem;margin-bottom:28px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:18px 22px;margin-bottom:14px;}}
.badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.8rem;font-weight:bold;}}
.bp{{background:rgba(52,211,153,.15);color:var(--green);border:1px solid var(--green);}}
.bf{{background:rgba(248,113,113,.15);color:var(--red);border:1px solid var(--red);}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-top:10px;}}
.stat-box{{background:var(--border);border-radius:6px;padding:12px 14px;}}
.stat-label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;}}
.stat-value{{font-size:1.5rem;font-weight:bold;color:var(--accent);margin-top:2px;}}
table{{width:100%;border-collapse:collapse;font-size:.86rem;margin-top:8px;}}
th{{background:var(--border);color:var(--accent);padding:7px 10px;text-align:left;
  font-size:.78rem;text-transform:uppercase;letter-spacing:.5px;}}
td{{padding:6px 10px;border-bottom:1px solid var(--border);}}
tr:last-child td{{border-bottom:none;}}
pre{{background:#111;border:1px solid var(--border);padding:12px;border-radius:6px;
  font-family:'Courier New',monospace;font-size:.83rem;color:#a5f3fc;overflow-x:auto;}}
footer{{color:var(--muted);font-size:.78rem;margin-top:48px;text-align:center;}}
hr{{border:none;border-top:1px solid var(--border);margin:28px 0;}}
</style></head><body><div class="container">
<h1>🔬 EFA / CFA Analysis Report</h1>
<p class="meta">Generated: {ts}</p>
"""

    # ── 1. Dataset Overview ──────────────────────────────────────
    html += f"""<h2>1. Dataset Overview</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="stat-label">Original Variables</div><div class="stat-value">{len(original_df.columns)}</div></div>
  <div class="stat-box"><div class="stat-label">After Cleaning</div><div class="stat-value">{len(cleaned_df.columns)}</div></div>
  <div class="stat-box"><div class="stat-label">Observations</div><div class="stat-value">{len(cleaned_df)}</div></div>
  <div class="stat-box"><div class="stat-label">Dropped</div><div class="stat-value">{len(dropped_vars)}</div></div>
</div>
<div class="card" style="margin-top:12px"><strong>Dropped Variables:</strong>
  <span style="color:var(--muted)">{', '.join(dropped_vars) if dropped_vars else 'None'}</span>
</div>
"""

    # ── 2. EFA Suitability ──────────────────────────────────────
    s = suitability
    kmo_b = "bp" if s["kmo_pass"] else "bf"
    bar_b = "bp" if s["bartlett_pass"] else "bf"
    ov_b  = "bp" if s["overall_pass"]  else "bf"
    _ov_txt  = "PASS" if s["overall_pass"]  else "FAIL"
    _kmo_txt = "PASS" if s["kmo_pass"]      else "FAIL"
    _bar_txt = "PASS" if s["bartlett_pass"]  else "FAIL"
    html += f"""<h2>2. EFA Suitability</h2>
<div class="card">
  <div style="margin-bottom:12px">Overall: <span class="badge {ov_b}">{_ov_txt}</span></div>
  <table><thead><tr><th>Test</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead><tbody>
  <tr><td>KMO</td><td>{s['kmo_model']} — <em>{s['kmo_label']}</em></td><td>≥ 0.60</td>
    <td><span class="badge {kmo_b}">{_kmo_txt}</span></td></tr>
  <tr><td>Bartlett's Sphericity</td><td>χ² = {s['bartlett_chi2']}, p = {s['bartlett_p']}</td><td>p &lt; 0.05</td>
    <td><span class="badge {bar_b}">{_bar_txt}</span></td></tr>
  </tbody></table>
</div>
"""

    # ── 3. EFA Results ──────────────────────────────────────────
    loadings = efa_result["loadings"]
    variance = efa_result["variance"]
    communalities = efa_result["communalities"]
    load_headers = "".join(f"<th>{col}</th>" for col in loadings.columns)
    load_rows = "".join(
        f"<tr><td>{v}</td>{''.join(f'<td>{loadings.loc[v,c]:.3f}</td>' for c in loadings.columns)}"
        f"<td style='color:{'#34d399' if communalities[v]>=.5 else ('#fbbf24' if communalities[v]>=.3 else '#f87171')}'>"
        f"{communalities[v]:.3f}</td></tr>"
        for v in loadings.index
    )
    var_rows = "".join(
        f"<tr><td>{idx}</td><td>{row['SS Loadings']:.4f}</td>"
        f"<td>{row['Proportion Var']*100:.2f}%</td><td>{row['Cumulative Var']*100:.2f}%</td></tr>"
        for idx, row in variance.iterrows()
    )
    diag_rows = "".join(
        f"<tr><td>{r['Variable']}</td><td>{r['MaxLoading']}</td><td>{r['FactorsAboveThreshold']}</td>"
        f"<td>{r['Communality']}</td><td>{r['Issue']}</td>"
        f"<td>{'✓ Drop' if r['RecommendDrop'] else '—'}</td></tr>"
        for _, r in diagnostics.iterrows()
    )
    html += f"""<h2>3. Exploratory Factor Analysis</h2>
<p style="color:var(--muted);margin-bottom:12px">Factors extracted: <strong style="color:var(--accent)">{efa_result['n_factors']}</strong></p>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Variable</th>{load_headers}<th>Communality</th></tr></thead><tbody>{load_rows}</tbody></table>
</div>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Factor</th><th>SS Loadings</th><th>Proportion Var</th><th>Cumulative Var</th></tr></thead>
  <tbody>{var_rows}</tbody></table>
</div>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Variable</th><th>Max Load</th><th># Factors ≥ Threshold</th>
    <th>Communality</th><th>Issue</th><th>Recommended</th></tr></thead>
  <tbody>{diag_rows}</tbody></table>
</div>
"""

    # ── 4. CFA Results ──────────────────────────────────────────
    if cfa_result and cfa_result["success"] and fit_assessment:
        fa = fit_assessment
        ov_b2 = "bp" if fa["overall_pass"] else "bf"
        fit_rows_parts = []
        for idx, d in fa["indices"].items():
            badge_cls  = "bp" if d["pass_"] else "bf"
            badge_text = "PASS" if d["pass_"] else "FAIL"
            fit_rows_parts.append(
                f"<tr><td>{idx}</td><td>{d['value']}</td>"
                f"<td>{d['direction']} {d['threshold']}</td>"
                f"<td><span class='badge {badge_cls}'>{badge_text}</span></td></tr>"
            )
        fit_rows = "".join(fit_rows_parts)
        _cfa_fit_txt = "ADEQUATE" if fa["overall_pass"] else "INADEQUATE"
        html += f"""<h2>4. Confirmatory Factor Analysis</h2>
<div class="card">
  <div style="margin-bottom:12px">
    Model Fit: <span class="badge {ov_b2}">{_cfa_fit_txt}</span>
    &nbsp;<span style="color:var(--muted);font-size:.85rem">{fa['n_pass']}/{fa['n_total']} indices passed</span>
  </div>
  <table><thead><tr><th>Index</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead>
  <tbody>{fit_rows}</tbody></table>
</div>
<h2 style="font-size:.95rem">Model Specification</h2>
<pre>{model_str}</pre>
"""
    else:
        err = cfa_result.get("error","") if cfa_result else "Not run"
        html += f"""<h2>4. Confirmatory Factor Analysis</h2>
<div class="card"><span class="badge bf">NOT COMPLETED</span>
  <p style="margin-top:8px;color:var(--muted)">{err}</p></div>
"""

    # ── 5. Synthetic Validation ─────────────────────────────────
    if synthetic_validation is not None:
        syn_rows = "".join(
            f"<tr><td>{r['Variable']}</td><td>{r['OrigMean']}</td><td>{r['SynMean']}</td>"
            f"<td>{r['OrigStd']}</td><td>{r['SynStd']}</td>"
            f"<td>{r['MeanDelta']}</td><td>{r['StdDelta']}</td></tr>"
            for _, r in synthetic_validation.iterrows()
        )
        html += f"""<h2>5. Synthetic Data Validation</h2>
<div class="card" style="overflow-x:auto">
  <table><thead><tr><th>Variable</th><th>Orig Mean</th><th>Syn Mean</th>
    <th>Orig Std</th><th>Syn Std</th><th>Mean Δ</th><th>Std Δ</th></tr></thead>
  <tbody>{syn_rows}</tbody></table>
</div>
"""

    html += """<hr>
<footer>EFA/CFA Analyser — Results should be interpreted in context of your research design and theoretical framework.</footer>
</div></body></html>"""
    return html


# ══════════════════════════════════════════════════════════════════
# ── SESSION STATE INIT ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

_DEFAULTS = dict(
    df_original=None, df_working=None, suitability=None,
    n_factors_auto=None, eigenvalues=None,
    efa_result=None, diagnostics=None, efa_done=False,
    cfa_result=None, fit_assessment=None,
    synthetic_factor=None, synthetic_corr=None,
    syn_validation=None, report_html=None,
)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if "dropped_vars" not in st.session_state:
    st.session_state["dropped_vars"] = []
S = st.session_state


# ══════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 EFA / CFA Analyser")
    st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:20px">Rigorous factor analysis pipeline</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ EFA Settings")
    loading_threshold = st.slider("Loading threshold", 0.30, 0.60, 0.40, 0.05,
        help="Minimum |loading| for a variable to be considered loaded on a factor.")
    communality_threshold = st.slider("Communality threshold", 0.20, 0.60, 0.30, 0.05,
        help="Variables below this are flagged for removal.")
    rotation_method = st.selectbox("Rotation method",
        ["varimax", "oblimin", "promax", "quartimax", "equamax"],
        help="Varimax = orthogonal. Oblimin/Promax = oblique (correlated factors).")

    st.divider()
    st.markdown("### 📐 CFA Thresholds")
    cfi_thresh  = st.slider("CFI ≥",   0.80, 0.99, 0.95, 0.01)
    tli_thresh  = st.slider("TLI ≥",   0.80, 0.99, 0.95, 0.01)
    rmsea_thresh = st.slider("RMSEA ≤", 0.04, 0.15, 0.06, 0.01)
    srmr_thresh  = st.slider("SRMR ≤",  0.04, 0.15, 0.08, 0.01)
    cfa_thresholds = dict(CFI=cfi_thresh, TLI=tli_thresh, RMSEA=rmsea_thresh, SRMR=srmr_thresh)

    st.divider()
    st.markdown("### 🧪 Synthetic Data")
    syn_n    = st.number_input("Sample size", 50, 10000, 500, 50)
    syn_seed = st.number_input("Random seed", 0, 9999, 42)

    st.divider()
    if st.button("🔄 Reset All", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# ── HEADER ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════

st.markdown("# 🔬 EFA / CFA Analyser")
st.markdown("_Upload a dataset and run a complete Exploratory and Confirmatory Factor Analysis pipeline. "
            "Diagnose issues, guide remediation, and export clean + synthetic datasets._")
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">1</span> Upload Dataset</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"],
        help="Numeric columns only. Rows with missing values are auto-dropped.")
with col_up2:
    st.markdown('<div class="info-box">💡 <b>Requirements</b><br>• Numeric variables only<br>• Minimum 5 variables<br>• Recommended ≥ 100 rows<br>• Missing values auto-dropped</div>', unsafe_allow_html=True)

if uploaded:
    _file_key = f"{uploaded.name}_{uploaded.size}"
    if S.get("_last_file_key") != _file_key:
        try:
            fname = uploaded.name.lower()
            if fname.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(uploaded, encoding="utf-8")
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    df_raw = pd.read_csv(uploaded, encoding="latin-1")
            elif fname.endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(uploaded)
            else:
                st.error("❌ Unsupported file type. Upload a .csv, .xlsx, or .xls file.")
                st.stop()
            df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()
            if len(df_numeric.columns) < 3:
                st.error("❌ Need at least 3 numeric variables. Non-numeric columns are excluded automatically.")
                st.stop()
            if len(df_numeric) < 30:
                st.warning("⚠️ Fewer than 30 observations — factor analysis results may be unreliable.")
            S.df_original    = df_numeric.copy()
            S.df_working     = df_numeric.copy()
            S.dropped_vars   = []
            S.suitability    = None
            S.efa_result     = None
            S.cfa_result     = None
            S.fit_assessment = None
            S.efa_done       = False
            S.synthetic_factor = None
            S.synthetic_corr   = None
            S.syn_validation   = None
            S.report_html      = None
            S["_last_file_key"] = _file_key
        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")
            st.stop()

if S.df_original is None:
    st.markdown('<div class="warn-box">👆 Upload a dataset to begin.</div>', unsafe_allow_html=True)
    st.stop()

with st.expander("📋 Dataset Preview", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, len(S.df_original), "Original Vars"),
        (c2, len(S.df_original.columns), "Variables"),
        (c3, len(S.df_working.columns), "Working Vars"),
        (c4, len(S.dropped_vars), "Dropped"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.dataframe(S.df_working.head(10), use_container_width=True)
    t1, t2 = st.tabs(["Descriptive Statistics", "Correlation Matrix"])
    with t1:
        st.dataframe(S.df_working.describe().T.round(4), use_container_width=True)
    with t2:
        st.plotly_chart(plot_correlation_matrix(S.df_working), use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 2 — EFA SUITABILITY
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">2</span> EFA Suitability Tests</div>', unsafe_allow_html=True)

if st.button("▶ Run Suitability Tests"):
    with st.spinner("Running KMO and Bartlett's tests…"):
        S.suitability = check_efa_suitability(S.df_working)
        ev_info = determine_n_factors(S.df_working)
        S.n_factors_auto = ev_info["suggested_n"]
        S.eigenvalues = ev_info["eigenvalues"]

if S.suitability:
    suit = S.suitability
    pill = "pill-pass" if suit["overall_pass"] else "pill-fail"
    text = "✓ SUITABLE FOR EFA" if suit["overall_pass"] else "✗ EFA SUITABILITY ISSUES"
    st.markdown(f'<b>Overall:</b> <span class="{pill}">{text}</span>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{suit["kmo_model"]}</div>'
                f'<div class="metric-label">KMO Score — {suit["kmo_label"]}</div>'
                f'<div style="margin-top:6px"><span class="{"pill-pass" if suit["kmo_pass"] else "pill-fail"}">{"PASS" if suit["kmo_pass"] else "FAIL"}</span></div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{suit["bartlett_p"]}</div>'
                f'<div class="metric-label">Bartlett p-value (χ²={suit["bartlett_chi2"]})</div>'
                f'<div style="margin-top:6px"><span class="{"pill-pass" if suit["bartlett_pass"] else "pill-fail"}">{"PASS" if suit["bartlett_pass"] else "FAIL"}</span></div></div>',
                unsafe_allow_html=True)

    if not suit["overall_pass"]:
        st.markdown('<div class="warn-box">⚠️ Suitability concerns detected. You can still proceed — results should be interpreted cautiously.</div>', unsafe_allow_html=True)

    st.markdown("#### Scree Plot")
    st.plotly_chart(plot_scree(S.eigenvalues, S.n_factors_auto), use_container_width=True)
    st.markdown(f'<div class="info-box">🔢 Kaiser criterion suggests <b>{S.n_factors_auto}</b> factor(s). Override below if theory or scree plot suggests otherwise.</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 3 — EFA
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">3</span> Exploratory Factor Analysis (EFA)</div>', unsafe_allow_html=True)

if S.suitability is None:
    st.markdown('<div class="warn-box">Complete Step 2 first.</div>', unsafe_allow_html=True)
else:
    n_factors_override = st.number_input(
        "Number of factors to extract", min_value=1,
        max_value=max(1, len(S.df_working.columns) - 1),
        value=S.n_factors_auto or 3, step=1,
        help="Kaiser-suggested by default. Override based on theory or scree plot elbow.",
    )

    if st.button("▶ Run EFA"):
        with st.spinner("Fitting factor model…"):
            S.efa_result = run_efa(S.df_working, n_factors=n_factors_override, rotation=rotation_method)
            S.diagnostics = diagnose_loadings(
                S.efa_result["loadings"], S.efa_result["communalities"],
                load_thresh=loading_threshold, comm_thresh=communality_threshold,
            )
            S.efa_done = True
            S.cfa_result = None

    if S.efa_done and S.efa_result:
        efa = S.efa_result
        t1, t2, t3 = st.tabs(["Factor Loadings", "Communalities", "Variance Explained"])
        with t1:
            st.plotly_chart(plot_loading_heatmap(efa["loadings"], threshold=loading_threshold), use_container_width=True)
            st.dataframe(efa["loadings"].round(3), use_container_width=True)
        with t2:
            st.plotly_chart(plot_communalities(efa["communalities"], comm_thresh=communality_threshold), use_container_width=True)
        with t3:
            st.dataframe(efa["variance"].round(4), use_container_width=True)
            total_var = efa["variance"]["Cumulative Var"].iloc[-1] * 100
            st.markdown(f'<div class="info-box">📊 Total variance explained: <b>{total_var:.1f}%</b></div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 4 — DIAGNOSTICS & REMEDIATION
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">4</span> Item Diagnostics & Remediation</div>', unsafe_allow_html=True)

if not S.efa_done or S.diagnostics is None:
    st.markdown('<div class="warn-box">Complete Step 3 first.</div>', unsafe_allow_html=True)
else:
    diag = S.diagnostics
    n_flagged = diag["RecommendDrop"].sum()
    st.markdown(f"**{n_flagged} variable(s) flagged** out of {len(diag)} — sorted by severity.")

    def _colour_issue(val):
        return ("color: #34d399" if val == "OK"
                else "color: #f87171" if "Cross" in str(val)
                else "color: #fbbf24" if "Weak" in str(val)
                else "color: #fb923c")

    styled = (diag.style
              .map(_colour_issue, subset=["Issue"])
              .format({"MaxLoading": "{:.3f}", "Communality": "{:.3f}", "Severity": "{:.3f}"})
              .background_gradient(subset=["Severity"], cmap="Reds"))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("#### Select Variables to Drop")
    st.markdown('<div class="info-box">📌 Pre-selected = app recommendations by severity. You have full control — add or remove any variable. EFA is re-run after each drop batch.</div>', unsafe_allow_html=True)

    default_drops = diag[diag["RecommendDrop"]]["Variable"].tolist()
    selected_drops = st.multiselect(
        "Variables to remove",
        options=S.df_working.columns.tolist(),
        default=[v for v in default_drops if v in S.df_working.columns],
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("▶ Apply Drops & Re-run EFA", disabled=(len(selected_drops) == 0)):
            with st.spinner("Dropping and re-fitting…"):
                cleaned = S.df_working.drop(columns=selected_drops, errors="ignore")
                S.df_working  = cleaned
                S.dropped_vars = list(set(S.dropped_vars + selected_drops))
                S.suitability  = check_efa_suitability(cleaned)
                ev_info = determine_n_factors(cleaned)
                S.n_factors_auto = ev_info["suggested_n"]
                S.eigenvalues = ev_info["eigenvalues"]
                S.efa_result   = run_efa(cleaned, n_factors=S.efa_result["n_factors"], rotation=rotation_method)
                S.diagnostics  = diagnose_loadings(
                    S.efa_result["loadings"], S.efa_result["communalities"],
                    load_thresh=loading_threshold, comm_thresh=communality_threshold,
                )
                S.cfa_result = None
                st.success(f"✓ Dropped {len(selected_drops)} variable(s). EFA re-run.")
                st.rerun()
    with col_d2:
        if st.button("▶ Proceed Without Drops"):
            st.markdown('<div class="info-box">✓ Proceeding with current variable set.</div>', unsafe_allow_html=True)

    if S.dropped_vars:
        st.markdown(f"**Total dropped so far:** `{', '.join(S.dropped_vars)}`")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 5 — CFA
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">5</span> Confirmatory Factor Analysis (CFA)</div>', unsafe_allow_html=True)

if not S.efa_done or S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    model_str, factor_vars = build_cfa_model(S.efa_result["loadings"], threshold=loading_threshold)
    st.markdown("#### CFA Model (auto-derived from EFA)")
    st.code(model_str, language="text")

    if not factor_vars:
        st.error("❌ No factors have ≥ 2 indicators. Lower the loading threshold or drop fewer variables.")
    else:
        st.markdown('<div class="info-box">ℹ️ Each item is assigned to its primary factor (highest |loading| ≥ threshold). This is an EFA-informed confirmatory re-specification — appropriate for academic validation.</div>', unsafe_allow_html=True)

        if st.button("▶ Run CFA"):
            with st.spinner("Fitting CFA model — this may take 10–30 seconds…"):
                S.cfa_result = run_cfa(S.df_working, model_str)
                if S.cfa_result["success"]:
                    S.fit_assessment = assess_cfa_fit(S.cfa_result["fit_indices"], cfa_thresholds)

    if S.cfa_result:
        if not S.cfa_result["success"]:
            st.error(f"❌ CFA failed: {S.cfa_result['error']}")
            st.markdown('<div class="warn-box">Common causes: too few observations, singular covariance matrix, or model identification issues. Try increasing observations, reducing factors, or dropping more items.</div>', unsafe_allow_html=True)
        else:
            fa = S.fit_assessment
            pill = "pill-pass" if fa["overall_pass"] else "pill-fail"
            text = "ADEQUATE FIT" if fa["overall_pass"] else "INADEQUATE FIT"
            st.markdown(f'<b>CFA Fit:</b> <span class="{pill}">{text}</span> '
                        f'<span style="color:var(--muted);font-size:.9rem">({fa["n_pass"]}/{fa["n_total"]} indices passed)</span>',
                        unsafe_allow_html=True)

            st.plotly_chart(plot_fit_indices(fa), use_container_width=True)

            fit_records = [dict(Index=idx, Value=d["value"],
                                Threshold=f"{d['direction']} {d['threshold']}",
                                Status="✓ PASS" if d["pass_"] else "✗ FAIL")
                           for idx, d in fa["indices"].items()]
            st.dataframe(pd.DataFrame(fit_records), use_container_width=True, hide_index=True)

            with st.expander("📊 Parameter Estimates"):
                if S.cfa_result["estimates"] is not None:
                    st.dataframe(S.cfa_result["estimates"], use_container_width=True)

            if not fa["overall_pass"]:
                st.markdown("#### 🔧 Modification Suggestions")
                for s in get_modification_suggestions(fa):
                    st.markdown(f'<div class="warn-box">⚠️ {s}</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">💡 Return to Step 4 to drop additional items, or Step 3 to re-extract with different n_factors, then re-run CFA.</div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 6 — SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">6</span> Synthetic Data Generation</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Factor-Structure Preserving")
        st.markdown('<div class="info-box">Simulates latent factor scores × loadings + unique variance. Preserves <b>psychometric structure</b>. Recommended for structural validity studies.</div>', unsafe_allow_html=True)
        if st.button("▶ Generate (Factor-Based)", use_container_width=True):
            with st.spinner("Simulating from factor structure…"):
                S.synthetic_factor = generate_factor_based(S.df_working, S.efa_result, n_samples=syn_n, seed=syn_seed)
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_factor)
            st.success(f"✓ {syn_n} synthetic observations generated.")
    with c2:
        st.markdown("#### Correlation Preserving")
        st.markdown('<div class="info-box">Multivariate normal from empirical <b>covariance matrix</b>. Faster, preserves pairwise correlations but not latent structure explicitly.</div>', unsafe_allow_html=True)
        if st.button("▶ Generate (Correlation-Based)", use_container_width=True):
            with st.spinner("Sampling from multivariate normal…"):
                S.synthetic_corr = generate_correlation_based(S.df_working, n_samples=syn_n, seed=syn_seed)
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_corr)
            st.success(f"✓ {syn_n} synthetic observations generated.")

    syn_display = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr
    if syn_display is not None:
        t1, t2, t3 = st.tabs(["Preview", "Validation Summary", "Distribution Comparison"])
        with t1:
            st.dataframe(syn_display.head(10).round(3), use_container_width=True)
        with t2:
            if S.syn_validation is not None:
                st.dataframe(S.syn_validation, use_container_width=True, hide_index=True)
        with t3:
            st.plotly_chart(plot_synthetic_comparison(S.df_working, syn_display), use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 7 — EXPORT
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="step-badge"><span class="step-num">7</span> Export Bundle</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) to enable exports.</div>', unsafe_allow_html=True)
else:
    syn_export = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr
    syn_label  = "factor_based" if S.synthetic_factor is not None else "correlation_based"

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.markdown("##### 🗃️ Cleaned Dataset")
        st.download_button(
            label=f"⬇ cleaned_data.csv\n({len(S.df_working)} rows × {len(S.df_working.columns)} cols)",
            data=S.df_working.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv", mime="text/csv", use_container_width=True,
        )

    with col_e2:
        st.markdown("##### 🧪 Synthetic Dataset")
        if syn_export is not None:
            st.download_button(
                label=f"⬇ synthetic_{syn_label}.csv\n({len(syn_export)} rows × {len(syn_export.columns)} cols)",
                data=syn_export.to_csv(index=False).encode("utf-8"),
                file_name=f"synthetic_{syn_label}.csv", mime="text/csv", use_container_width=True,
            )
        else:
            st.markdown('<div class="warn-box">Generate synthetic data in Step 6 first.</div>', unsafe_allow_html=True)

    with col_e3:
        st.markdown("##### 📄 Analysis Report")
        has_cfa = S.cfa_result is not None and S.cfa_result["success"]
        if has_cfa:
            if st.button("🔨 Build HTML Report", use_container_width=True):
                with st.spinner("Compiling report…"):
                    S.report_html = generate_html_report(
                        original_df=S.df_original, cleaned_df=S.df_working,
                        suitability=S.suitability, efa_result=S.efa_result,
                        diagnostics=S.diagnostics, dropped_vars=S.dropped_vars,
                        cfa_result=S.cfa_result, fit_assessment=S.fit_assessment,
                        cfa_thresholds=cfa_thresholds, synthetic_validation=S.syn_validation,
                        model_str=S.cfa_result.get("model_str",""),
                    )
                st.success("✓ Report ready.")
            if S.report_html:
                st.download_button(
                    label="⬇ efa_cfa_report.html",
                    data=S.report_html.encode("utf-8"),
                    file_name="efa_cfa_report.html", mime="text/html", use_container_width=True,
                )
        else:
            st.markdown('<div class="warn-box">Run CFA (Step 5) to enable report generation.</div>', unsafe_allow_html=True)

    # ── Full ZIP ─────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 📦 Download Everything as ZIP")
    if st.button("⬇ Build & Download Full Bundle (.zip)"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("cleaned_data.csv", S.df_working.to_csv(index=False))
            if syn_export is not None:
                zf.writestr(f"synthetic_{syn_label}.csv", syn_export.to_csv(index=False))
            if S.report_html:
                zf.writestr("efa_cfa_report.html", S.report_html)
            if S.cfa_result and S.cfa_result.get("model_str"):
                zf.writestr("cfa_model.txt", S.cfa_result["model_str"])
            if S.efa_result:
                zf.writestr("efa_loadings.csv", S.efa_result["loadings"].round(4).to_csv())
                zf.writestr("efa_communalities.csv", S.efa_result["communalities"].round(4).to_frame().to_csv())
        zip_buffer.seek(0)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="⬇ efa_cfa_bundle.zip",
            data=zip_buffer.getvalue(),
            file_name=f"efa_cfa_bundle_{ts_str}.zip",
            mime="application/zip",
        )


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#475569;font-size:.82rem">'
            'EFA/CFA Analyser &nbsp;|&nbsp; Built for researchers. '
            'Interpret results within your theoretical framework and research design.'
            '</div>', unsafe_allow_html=True)
