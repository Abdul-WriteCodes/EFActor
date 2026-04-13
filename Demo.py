"""
DataVizX — No-Code Chart Builder (Demo / Test Version)
=======================================================
Auth: Single password via st.secrets["DVX_PASSWORD"] (no Google Sheets).
No credit deduction — all exports are free in this build.
Swap in the full datavizx.py when ready for production.
"""

import io
import base64
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

CHART_TYPES = [
    "Bar",
    "Line",
    "Scatter",
    "Heatmap / Correlation",
    "Box Plot",
    "Pie / Donut",
    "Histogram",
]

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        colorway=["#00C2A8", "#7B6CF6", "#F59E0B", "#F43F5E", "#10B981",
                  "#38bdf8", "#e879f9", "#fb923c"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        title=dict(font=dict(family="Outfit, sans-serif", size=18, color="#f0f4ff")),
    )
)


# ═══════════════════════════════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════════════════════════════

def _check_password(pw: str) -> bool:
    stored = st.secrets.get("DVX_PASSWORD", "datavizx2024")
    return hashlib.sha256(pw.encode()).hexdigest() == \
           hashlib.sha256(stored.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════

def inject_css():
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg:       #080b12;
        --surface:  #0d1117;
        --surface2: #111827;
        --surface3: #1a2332;
        --border:   rgba(255,255,255,0.07);
        --border2:  rgba(255,255,255,0.12);
        --text:     #f0f4ff;
        --muted:    #64748b;
        --teal:     #00C2A8;
        --violet:   #7B6CF6;
        --amber:    #F59E0B;
        --rose:     #F43F5E;
        --green:    #10B981;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 4px; }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .main .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1200px !important;
    }

    #MainMenu, footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"] > div:first-child { visibility: hidden; }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    button[kind="header"],
    [data-testid="stHeader"] button {
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: all !important;
        z-index: 999999 !important;
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'Outfit', sans-serif !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(0,194,168,0.15) !important;
        outline: none !important;
    }
    .stTextInput label, .stNumberInput label,
    .stSelectbox label, .stRadio label {
        color: var(--muted) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stSelectbox > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--teal), #00a896) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(0,194,168,0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,194,168,0.35) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    .stRadio > div { gap: 0.3rem !important; }
    .stRadio > div > label {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.9rem !important;
        cursor: pointer !important;
        transition: all 0.15s !important;
        color: var(--muted) !important;
        font-size: 0.88rem !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    .stRadio > div > label:hover {
        border-color: var(--teal) !important;
        color: var(--text) !important;
    }

    [data-testid="stForm"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }

    .stSpinner > div { border-top-color: var(--teal) !important; }
    hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 8px rgba(0,194,168,0.3); }
        50%       { box-shadow: 0 0 20px rgba(0,194,168,0.6); }
    }
    .fade-in   { animation: fadeSlideUp 0.45s ease forwards; }
    .fade-in-2 { animation: fadeSlideUp 0.45s ease 0.1s both; }
    .fade-in-3 { animation: fadeSlideUp 0.45s ease 0.2s both; }
    </style>
    """)


# ═══════════════════════════════════════════════════════════════════════════
#  LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════════════

def login_screen():
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.html("""
        <div class="fade-in" style="padding-top:7vh;text-align:center;">
            <div style="
                width:72px;height:72px;border-radius:20px;margin:0 auto 1.5rem;
                background:linear-gradient(135deg,#00C2A8,#7B6CF6);
                display:flex;align-items:center;justify-content:center;
                font-size:2rem;box-shadow:0 8px 32px rgba(0,194,168,0.35);
                animation:pulse-glow 3s ease-in-out infinite;
            ">📊</div>
            <div style="
                font-family:'Outfit',sans-serif;font-size:2.6rem;font-weight:900;
                letter-spacing:-1.5px;line-height:1;margin-bottom:0.4rem;
                background:linear-gradient(135deg,#f0f4ff 30%,#00C2A8);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            ">DataVizX</div>
            <div style="
                font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                color:#64748b;letter-spacing:0.2em;text-transform:uppercase;
                margin-bottom:2rem;
            ">No-Code Chart Builder &middot; Demo Build</div>
            <div style="
                font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                color:#64748b;letter-spacing:0.15em;text-transform:uppercase;
                margin-bottom:0.8rem;text-align:left;
            ">&#128274; Password</div>
        </div>
        """)

        pw = st.text_input("Password", type="password",
                           placeholder="Enter password…",
                           label_visibility="collapsed")
        if st.button("Enter DataVizX →", use_container_width=True, type="primary"):
            if not pw:
                st.error("Please enter the password.")
            elif _check_password(pw):
                st.session_state["dvx_auth"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")

        st.html("""
        <div style="
            font-family:'JetBrains Mono',monospace;font-size:0.68rem;
            color:#1e293b;text-align:center;margin-top:1.5rem;
        ">Demo build &middot; All exports free &middot; No credit tracking</div>
        """)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.html("""
        <div style="
            padding:1.8rem 1rem 1.2rem;
            border-bottom:1px solid rgba(255,255,255,0.06);
            margin-bottom:1.2rem;
        ">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
                <div style="
                    width:34px;height:34px;border-radius:9px;
                    background:linear-gradient(135deg,#00C2A8,#7B6CF6);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1rem;flex-shrink:0;
                ">📊</div>
                <span style="
                    font-family:'Outfit',sans-serif;font-size:1.25rem;
                    font-weight:800;letter-spacing:-0.5px;color:#f0f4ff;
                ">DataVizX</span>
            </div>
            <div style="
                font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#334155;letter-spacing:0.15em;text-transform:uppercase;
                padding-left:44px;
            ">Demo Build</div>
        </div>
        """)

        st.html("""
        <div style="
            background:rgba(0,194,168,0.08);border:1px solid rgba(0,194,168,0.2);
            border-radius:10px;padding:0.7rem 1rem;margin-bottom:1.4rem;
            font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#00C2A8;
        ">⚡ Demo mode — exports are free</div>
        """)

        st.html("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
            color:#475569;letter-spacing:0.15em;text-transform:uppercase;
            margin-bottom:0.5rem;padding:0 0.2rem;">Chart Type</div>
        """)

        chart_type = st.radio(
            "chart_type",
            CHART_TYPES,
            label_visibility="collapsed",
        )

        st.html("<br>" * 3)
        st.divider()

        if st.button("Sign Out", use_container_width=True):
            st.session_state["dvx_auth"] = False
            st.rerun()

        st.html("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
            color:#1e293b;text-align:center;margin-top:0.8rem;">
            v1.0-demo · DataVizX
        </div>
        """)

    return chart_type


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════

def data_uploader() -> pd.DataFrame | None:
    st.html("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
        color:#475569;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem;">
        1 · Load Data
    </div>
    """)

    input_method = st.radio(
        "input_method",
        ["Upload CSV", "Paste / type data", "Use sample dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    df = None

    if input_method == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅  Loaded {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    elif input_method == "Paste / type data":
        raw = st.text_area(
            "Paste CSV data (first row = headers)",
            height=180,
            placeholder="Month,Sales,Leads\nJan,120,200\nFeb,140,230\n...",
            label_visibility="collapsed",
        )
        if raw.strip():
            try:
                df = pd.read_csv(io.StringIO(raw))
                st.success(f"✅  Parsed {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not parse data: {e}")

    else:
        sample_choice = st.selectbox(
            "Choose a sample dataset",
            ["Monthly Sales", "Product Performance", "Survey Responses", "Stock Prices"],
            label_visibility="collapsed",
        )
        df = _get_sample(sample_choice)
        st.info(f"📋 Using sample: **{sample_choice}** — {len(df)} rows × {len(df.columns)} columns")

    if df is not None:
        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    return df


def _get_sample(name: str) -> pd.DataFrame:
    if name == "Monthly Sales":
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        return pd.DataFrame({
            "Month":   months,
            "Revenue": [12400,13800,11200,15600,17800,16200,19400,21000,18600,22400,24800,28600],
            "Leads":   [200,230,190,260,310,290,340,380,320,400,450,520],
            "Signups": [42,51,38,67,85,72,94,108,89,121,139,168],
        })
    elif name == "Product Performance":
        return pd.DataFrame({
            "Product":   ["DataSynthX","PanelStatX","DataVizX","SplitStatX","CleanStatX"],
            "Revenue":   [8400,6200,3800,1200,900],
            "Users":     [142,98,61,18,14],
            "Churn_pct": [3.2,4.1,2.8,5.5,6.2],
            "NPS":       [72,68,81,55,49],
        })
    elif name == "Survey Responses":
        np.random.seed(42)
        n = 120
        return pd.DataFrame({
            "Age":          np.random.choice([18,25,35,45,55,65], n),
            "Satisfaction": np.random.randint(1, 6, n),
            "Would_Refer":  np.random.choice(["Yes","No","Maybe"], n),
            "Usage_hrs":    np.round(np.random.exponential(4, n), 1),
            "Plan":         np.random.choice(["Free","Pro","Enterprise"], n, p=[0.5,0.35,0.15]),
        })
    else:
        dates = pd.date_range("2024-01-01", periods=52, freq="W")
        np.random.seed(7)
        return pd.DataFrame({
            "Week":   dates.strftime("%Y-%m-%d"),
            "StockA": np.round(100 + np.cumsum(np.random.randn(52) * 2), 2),
            "StockB": np.round(80  + np.cumsum(np.random.randn(52) * 1.5), 2),
            "Volume": np.random.randint(10000, 50000, 52),
        })


# ═══════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _apply_theme(fig: go.Figure, title: str, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title_text=title,
        showlegend=show_legend,
        margin=dict(t=60, b=40, l=40, r=40),
        height=520,
    )
    return fig


def build_bar(df):
    cols     = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    c1, c2, c3 = st.columns(3)
    x_col    = c1.selectbox("X axis (category)", cols)
    y_cols   = c2.multiselect("Y axis (values)", num_cols, default=num_cols[:1])
    bar_mode = c3.selectbox("Mode", ["group", "stack", "relative"])

    color_col = None
    if cat_cols:
        cc = st.selectbox("Color by (optional)", ["None"] + cat_cols)
        color_col = None if cc == "None" else cc

    title = st.text_input("Chart title", "Bar Chart", key="bar_title")

    if not y_cols:
        st.warning("Select at least one Y column.")
        return None

    if len(y_cols) == 1:
        fig = px.bar(df, x=x_col, y=y_cols[0], color=color_col, barmode=bar_mode,
                     color_discrete_sequence=PLOTLY_TEMPLATE["layout"]["colorway"])
    else:
        fig = go.Figure()
        for i, yc in enumerate(y_cols):
            fig.add_trace(go.Bar(
                name=yc, x=df[x_col], y=df[yc],
                marker_color=PLOTLY_TEMPLATE["layout"]["colorway"][i % 8],
            ))
        fig.update_layout(barmode=bar_mode)

    return _apply_theme(fig, title)


def build_line(df):
    cols     = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    c1, c2 = st.columns(2)
    x_col  = c1.selectbox("X axis", cols)
    y_cols = c2.multiselect("Y axis (one or more series)", num_cols, default=num_cols[:2])

    c3, c4 = st.columns(2)
    markers = c3.checkbox("Show markers", value=True)
    area    = c4.checkbox("Area fill", value=False)
    title   = st.text_input("Chart title", "Line Chart", key="line_title")

    if not y_cols:
        st.warning("Select at least one Y column.")
        return None

    mode = "lines+markers" if markers else "lines"
    fig  = go.Figure()
    colors = PLOTLY_TEMPLATE["layout"]["colorway"]
    for i, yc in enumerate(y_cols):
        kwargs = dict(
            name=yc, x=df[x_col], y=df[yc], mode=mode,
            line=dict(color=colors[i % 8], width=2.5),
            marker=dict(size=6),
        )
        if area:
            kwargs["fill"]      = "tozeroy" if i == 0 else "tonexty"
            kwargs["fillcolor"] = colors[i % 8] + "20"
        fig.add_trace(go.Scatter(**kwargs))

    return _apply_theme(fig, title)


def build_scatter(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    c1, c2 = st.columns(2)
    x_col  = c1.selectbox("X axis", num_cols)
    y_col  = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))

    c3, c4 = st.columns(2)
    color_col = c3.selectbox("Color by (optional)", ["None"] + all_cols)
    size_col  = c4.selectbox("Size by (optional)",  ["None"] + num_cols)
    regression = st.checkbox("Show regression line", value=True)
    title      = st.text_input("Chart title", "Scatter Plot", key="scatter_title")

    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=None if color_col == "None" else color_col,
        size=None  if size_col  == "None" else size_col,
        trendline="ols" if regression else None,
        color_discrete_sequence=PLOTLY_TEMPLATE["layout"]["colorway"],
    )
    fig.update_traces(marker=dict(opacity=0.8))
    return _apply_theme(fig, title)


def build_heatmap(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        return None

    selected = st.multiselect("Columns to include", num_cols, default=num_cols)
    title    = st.text_input("Chart title", "Correlation Matrix", key="heat_title")

    if len(selected) < 2:
        st.warning("Select at least 2 columns.")
        return None

    corr = df[selected].corr()
    fig  = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0,"#F43F5E"],[0.5,"#111827"],[1,"#00C2A8"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono"),
        hoverongaps=False,
    ))
    return _apply_theme(fig, title, show_legend=False)


def build_box(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    c1, c2 = st.columns(2)
    y_cols   = c1.multiselect("Value columns", num_cols, default=num_cols[:2])
    group_by = c2.selectbox("Group by (optional)", ["None"] + cat_cols)

    points = st.selectbox("Show points", ["outliers", "all", "none"])
    title  = st.text_input("Chart title", "Box Plot", key="box_title")

    if not y_cols:
        st.warning("Select at least one value column.")
        return None

    colors = PLOTLY_TEMPLATE["layout"]["colorway"]
    fig = go.Figure()
    for i, yc in enumerate(y_cols):
        if group_by != "None":
            for j, grp in enumerate(df[group_by].unique()):
                fig.add_trace(go.Box(
                    y=df[df[group_by] == grp][yc],
                    name=f"{yc} — {grp}",
                    marker_color=colors[(i * 3 + j) % 8],
                    boxpoints=points if points != "none" else False,
                    jitter=0.3, pointpos=0,
                ))
        else:
            fig.add_trace(go.Box(
                y=df[yc], name=yc,
                marker_color=colors[i % 8],
                boxpoints=points if points != "none" else False,
                jitter=0.3, pointpos=0,
            ))

    return _apply_theme(fig, title)


def build_pie(df):
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    c1, c2 = st.columns(2)
    label_col = c1.selectbox("Labels column", all_cols)
    value_col = c2.selectbox("Values column", num_cols)

    c3, c4 = st.columns(2)
    is_donut = c3.checkbox("Donut style", value=True)
    show_pct = c4.checkbox("Show percentages", value=True)
    title    = st.text_input("Chart title", "Pie Chart", key="pie_title")

    fig = go.Figure(data=go.Pie(
        labels=df[label_col],
        values=df[value_col],
        hole=0.45 if is_donut else 0,
        textinfo="label+percent" if show_pct else "label",
        marker=dict(
            colors=PLOTLY_TEMPLATE["layout"]["colorway"],
            line=dict(color="#080b12", width=2),
        ),
        textfont=dict(family="JetBrains Mono", size=11),
    ))
    return _apply_theme(fig, title)


def build_histogram(df):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    c1, c2 = st.columns(2)
    col   = c1.selectbox("Column", num_cols)
    nbins = c2.slider("Number of bins", 5, 100, 20)

    c3, c4 = st.columns(2)
    color_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols)
    kde       = c4.checkbox("Overlay KDE curve", value=True)
    title     = st.text_input("Chart title", "Histogram", key="hist_title")

    fig = px.histogram(
        df, x=col, nbins=nbins,
        color=None if color_col == "None" else color_col,
        color_discrete_sequence=PLOTLY_TEMPLATE["layout"]["colorway"],
        opacity=0.85,
    )

    if kde and color_col == "None":
        try:
            from scipy.stats import gaussian_kde
            vals = df[col].dropna().values
            if len(vals) > 5:
                kde_x = np.linspace(vals.min(), vals.max(), 300)
                kde_y = gaussian_kde(vals)(kde_x)
                bin_width = (vals.max() - vals.min()) / nbins
                fig.add_trace(go.Scatter(
                    x=kde_x, y=kde_y * len(vals) * bin_width,
                    mode="lines", name="KDE",
                    line=dict(color="#7B6CF6", width=2.5, dash="dot"),
                ))
        except ImportError:
            pass

    return _apply_theme(fig, title)


CHART_BUILDERS = {
    "Bar":                   build_bar,
    "Line":                  build_line,
    "Scatter":               build_scatter,
    "Heatmap / Correlation": build_heatmap,
    "Box Plot":              build_box,
    "Pie / Donut":           build_pie,
    "Histogram":             build_histogram,
}


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT PANEL
# ═══════════════════════════════════════════════════════════════════════════

def export_panel(fig: go.Figure):
    st.html("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
        color:#475569;text-transform:uppercase;letter-spacing:0.12em;
        margin-top:1.5rem;margin-bottom:0.8rem;">
        3 · Export
    </div>
    """)

    c1, c2 = st.columns(2)

    with c1:
        scale = st.select_slider(
            "PNG resolution", options=[1, 2, 3], value=2,
            format_func=lambda x: {1:"1× Standard", 2:"2× HiRes", 3:"3× Print"}[x],
        )
        if st.button("⬇️  Export PNG", use_container_width=True):
            try:
                img_bytes = fig.to_image(format="png", scale=scale, width=1200, height=520*scale)
                st.download_button(
                    label="📥 Click to Download PNG",
                    data=img_bytes,
                    file_name=f"datavizx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except ImportError:
                st.error("Run `pip install kaleido` to enable PNG export.")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with c2:
        st.html("<div style='height:2.4rem'></div>")
        try:
            img_b64 = base64.b64encode(
                fig.to_image(format="png", scale=2, width=1200, height=1040)
            ).decode()
            st.html(f"""
            <button onclick="
                fetch('data:image/png;base64,{img_b64}')
                    .then(r=>r.blob())
                    .then(b=>navigator.clipboard.write([new ClipboardItem({{'image/png':b}})]));
                this.textContent='✅ Copied!';
                setTimeout(()=>this.textContent='📋  Copy to Clipboard',2000);
            " style="
                width:100%;padding:0.6rem 1rem;
                background:linear-gradient(135deg,#7B6CF6,#6357d4);
                color:#fff;border:none;border-radius:10px;
                font-family:'Outfit',sans-serif;font-weight:700;font-size:0.9rem;
                cursor:pointer;box-shadow:0 4px 15px rgba(123,108,246,0.25);
            ">📋  Copy to Clipboard</button>
            """)
        except Exception:
            st.info("Install kaleido to enable clipboard copy.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APP PAGE
# ═══════════════════════════════════════════════════════════════════════════

def app_page(chart_type: str):
    now  = datetime.now(timezone.utc)
    hour = now.hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    st.html(f"""
    <div class="fade-in" style="margin-bottom:1.8rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
            color:#475569;letter-spacing:0.15em;text-transform:uppercase;
            margin-bottom:0.3rem;">
            {greeting} — {now.strftime("%A, %d %b %Y")}
        </div>
        <h1 style="
            font-family:'Outfit',sans-serif;font-size:2rem;font-weight:800;
            letter-spacing:-1px;margin:0;line-height:1.1;
            background:linear-gradient(135deg,#f0f4ff 40%,#00C2A8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        ">{chart_type}</h1>
        <p style="color:#475569;font-size:0.88rem;margin-top:0.3rem;">
            Upload your data, configure the chart, then export.
        </p>
    </div>
    """)

    df = data_uploader()

    if df is None or df.empty:
        st.html("""
        <div style="
            background:#0d1117;border:1px dashed rgba(255,255,255,0.1);
            border-radius:16px;padding:3rem;text-align:center;margin-top:1rem;
        ">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">📂</div>
            <div style="font-family:'Outfit',sans-serif;font-size:1rem;color:#475569;">
                Load data above to start building your chart
            </div>
        </div>
        """)
        return

    st.html("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
        color:#475569;text-transform:uppercase;letter-spacing:0.12em;
        margin-top:1.5rem;margin-bottom:0.8rem;">
        2 · Configure Chart
    </div>
    """)

    fig = CHART_BUILDERS[chart_type](df)

    if fig is not None:
        st.html("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
            color:#475569;text-transform:uppercase;letter-spacing:0.12em;
            margin-top:1.5rem;margin-bottom:0.5rem;">
            Preview
        </div>
        """)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        export_panel(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="DataVizX — Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    if not st.session_state.get("dvx_auth"):
        login_screen()
        return

    chart_type = render_sidebar()
    app_page(chart_type)


if __name__ == "__main__":
    main()
