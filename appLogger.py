"""
╔══════════════════════════════════════════════════════════════════╗
║                  EFActor  —  Single-File Streamlit App           ║
║         EFA / CFA Analysis Platform  |  efactor.app              ║
║  Landing → Auth → Upload → EFA → CFA → Diagnose → Export        ║
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
import gspread
from google.oauth2.service_account import Credentials

# ── sklearn >= 1.6 compatibility ─────────────────────────────────────────────
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
    page_title="EFActor — Psychometric Analysis Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════════
C = dict(
    accent="#6c8dfa", accent2="#a78bfa",
    green="#34d399", red="#f87171", yellow="#fbbf24",
    bg="#0a0c14", surface="#111520", surface2="#181d2e", border="#1e2540",
    text="#e2e8f0", muted="#64748b",
)
LAYOUT_BASE = dict(
    paper_bgcolor=C["surface"], plot_bgcolor=C["surface"],
    font=dict(color=C["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[C["accent"], C["accent2"], C["green"], C["yellow"], C["red"]],
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --accent:#6c8dfa; --accent2:#a78bfa; --green:#34d399;
  --red:#f87171; --yellow:#fbbf24; --border:#1e2540;
  --surface:#111520; --surface2:#181d2e; --muted:#64748b;
  --bg:#0a0c14; --text:#e2e8f0;
}
html,body,[class*="css"]{ font-family:'Inter',system-ui,sans-serif; background-color:var(--bg); }
.stApp{ background:var(--bg); }
[data-testid="stSidebar"]{ background:#0d1020 !important; border-right:1px solid var(--border); }
h1{ color:var(--accent) !important; font-weight:800 !important; letter-spacing:-0.5px; }
h2{ color:var(--accent2) !important; border-bottom:1px solid var(--border); padding-bottom:8px; font-weight:700 !important; }
h3{ color:var(--text) !important; font-weight:600 !important; }
.step-badge{ display:inline-flex; align-items:center; gap:10px;
  background:linear-gradient(135deg,var(--surface),var(--surface2));
  border:1px solid var(--border); border-left:3px solid var(--accent);
  padding:11px 18px; border-radius:8px; color:var(--text);
  font-size:.95rem; font-weight:600; margin-bottom:18px; width:100%; }
.step-num{ background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:var(--bg); border-radius:50%; width:26px; height:26px;
  display:inline-flex; align-items:center; justify-content:center;
  font-size:.8rem; font-weight:700; flex-shrink:0; }
.pill-pass{ background:rgba(52,211,153,.12); color:var(--green); border:1px solid rgba(52,211,153,.4);
  border-radius:20px; padding:3px 14px; font-size:.78rem; font-weight:600; display:inline-block; }
.pill-fail{ background:rgba(248,113,113,.12); color:var(--red); border:1px solid rgba(248,113,113,.4);
  border-radius:20px; padding:3px 14px; font-size:.78rem; font-weight:600; display:inline-block; }
.metric-card{ background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:18px 20px; text-align:center; position:relative; overflow:hidden; }
.metric-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.metric-val{ font-size:2rem; font-weight:700; color:var(--accent); line-height:1.1; }
.metric-label{ font-size:.7rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:1px; margin-top:6px; font-weight:500; }
.info-box{ background:rgba(108,141,250,.07); border:1px solid rgba(108,141,250,.2);
  border-radius:8px; padding:12px 16px; color:#c7d2fe; font-size:.88rem; margin:8px 0; line-height:1.6; }
.warn-box{ background:rgba(251,191,36,.07); border:1px solid rgba(251,191,36,.25);
  border-radius:8px; padding:12px 16px; color:#fde68a; font-size:.88rem; margin:8px 0; line-height:1.6; }
.lock-box{ background:rgba(108,141,250,.06); border:1px solid rgba(108,141,250,.2);
  border-left:3px solid var(--accent); border-radius:8px; padding:18px 20px; margin:8px 0; }
.section-divider{ border:none; border-top:1px solid var(--border); margin:32px 0; }
code,pre{ font-family:'JetBrains Mono',monospace !important; }
.stButton>button{ background:linear-gradient(135deg,#151a2e,#1e2745) !important;
  border:1px solid var(--accent) !important; color:var(--accent) !important;
  font-family:'Inter',sans-serif !important; font-size:.92rem !important;
  font-weight:600 !important; border-radius:8px !important; }
.stButton>button:hover{ background:linear-gradient(135deg,var(--accent),#4f6fe8) !important;
  color:var(--bg) !important; border-color:transparent !important; }
.stDownloadButton>button{ background:linear-gradient(135deg,#0f1f18,#122518) !important;
  border:1px solid var(--green) !important; color:var(--green) !important;
  border-radius:8px !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab-list"]{ background:var(--surface); border-radius:10px;
  padding:4px; gap:4px; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ background:transparent; border-radius:8px; color:var(--muted);
  font-size:13px; font-weight:500; padding:8px 18px; border:none; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,var(--accent),#4f6fe8) !important; color:white !important; }
.stTextInput input,.stNumberInput input{ background:var(--surface2) !important;
  border:1px solid var(--border) !important; border-radius:8px !important; color:var(--text) !important; }
#MainMenu{ visibility:hidden; } footer{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS / CREDIT ENGINE
# ══════════════════════════════════════════════════════════════════
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_NAME       = "Sheet1"
COL_KEY          = "Key"
COL_CREDITS      = "Credits"
COL_OWNER        = "Email"
COL_ISSUED       = "DatePurchased"
REQUIRED_HEADERS = [COL_KEY, COL_CREDITS, COL_ISSUED, COL_OWNER]


@st.cache_resource(show_spinner=False)
def _get_gsheet_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def _get_keys_worksheet():
    gc = _get_gsheet_client()
    sheet_id = st.secrets["EFACTOR_SHEET_ID"]
    sh = gc.open_by_key(sheet_id)
    return sh.worksheet(SHEET_NAME)


def validate_key(access_key: str):
    try:
        ws = _get_keys_worksheet()
        records = ws.get_all_records(
            expected_headers=REQUIRED_HEADERS,
            value_render_option="UNFORMATTED_VALUE",
        )
        records = [r for r in records if any(str(v).strip() for v in r.values())]
        for row in records:
            if str(row.get(COL_KEY, "")).strip() == access_key.strip():
                return row
        return None
    except Exception as e:
        st.error(f"Key validation error: {e}")
        return None


def get_credits(access_key: str) -> int:
    try:
        ws = _get_keys_worksheet()
        records = ws.get_all_records(
            expected_headers=REQUIRED_HEADERS,
            value_render_option="UNFORMATTED_VALUE",
        )
        records = [r for r in records if any(str(v).strip() for v in r.values())]
        for row in records:
            if str(row.get(COL_KEY, "")).strip() == access_key.strip():
                return int(row.get(COL_CREDITS, 0))
        return 0
    except Exception:
        return 0


def deduct_credits(access_key: str, amount: int) -> int:
    try:
        ws = _get_keys_worksheet()
        records = ws.get_all_records(
            expected_headers=REQUIRED_HEADERS,
            value_render_option="UNFORMATTED_VALUE",
        )
        records = [r for r in records if any(str(v).strip() for v in r.values())]
        header = ws.row_values(1)
        credits_col_idx = header.index(COL_CREDITS) + 1
        for i, row in enumerate(records):
            if str(row.get(COL_KEY, "")).strip() == access_key.strip():
                row_number = i + 2
                current  = int(row.get(COL_CREDITS, 0))
                new_val  = max(0, current - amount)
                ws.update_cell(row_number, credits_col_idx, new_val)
                return new_val
        return 0
    except Exception as e:
        st.error(f"Credit deduction error: {e}")
        return 0


def log_event(access_key: str, event: str, metadata: str = ""):
    """Log a funnel event to the Events sheet. Never raises."""
    try:
        gc = _get_gsheet_client()
        sh = gc.open_by_key(st.secrets["EFACTOR_SHEET_ID"])
        try:
            ws = sh.worksheet("Events")
        except Exception:
            ws = sh.add_worksheet(title="Events", rows=1000, cols=4)
            ws.append_row(["Timestamp", "AccessKey", "Event", "Metadata"])
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            access_key or "trial",
            event,
            metadata,
        ])
    except Exception:
        pass  # silently swallow — logging must never crash the app


def export_credit_cost(n_rows: int) -> int:
    if n_rows <= 300:
        return 1
    elif n_rows <= 1000:
        return 2
    else:
        return 5


# ══════════════════════════════════════════════════════════════════
# PLANS CONFIG
# ══════════════════════════════════════════════════════════════════
PLANS = [
    {
        "name": "Starter", "price": "$9", "credits": "10 credits",
        "period": "one-time", "color": "#34d399", "highlight": False,
        "features": ["Unlimited EFA & CFA runs","10 dataset exports",
                     "Up to 1,000 rows per export","Synthetic data generation",
                     "Word analysis report (.docx)","Access key via email"],
        "link": "https://your-payment-link-starter",
    },
    {
        "name": "Researcher", "price": "$19", "credits": "30 credits",
        "period": "one-time", "color": "#6c8dfa", "highlight": True,
        "features": ["Unlimited EFA & CFA runs","30 dataset exports",
                     "Up to 5,000 rows per export","Synthetic data generation",
                     "Word analysis report (.docx)","ZIP bundle export","Access key via email"],
        "link": "https://your-payment-link-researcher",
    },
    {
        "name": "Pro", "price": "$49", "credits": "100 credits",
        "period": "one-time", "color": "#a78bfa", "highlight": False,
        "features": ["Unlimited EFA & CFA runs","100 dataset exports",
                     "Unlimited rows per export","Synthetic data generation",
                     "Word analysis report (.docx)","ZIP bundle export",
                     "Priority support","Access key via email"],
        "link": "https://your-payment-link-pro",
    },
]


# ══════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════
def render_landing():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]{ display:none !important; }
    .block-container{ max-width:1020px; margin:0 auto; padding-top:0rem; }
    </style>""", unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="text-align:center;padding:72px 20px 48px;">
      <div style="display:inline-flex;align-items:center;gap:14px;margin-bottom:32px;">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#6c8dfa,#a78bfa);
                    border-radius:10px;display:flex;align-items:center;justify-content:center;
                    font-size:22px;flex-shrink:0;">◈</div>
        <span style="font-family:Inter,sans-serif;font-size:26px;font-weight:800;
                     background:linear-gradient(90deg,#6c8dfa,#a78bfa);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     letter-spacing:-0.5px;">EFActor</span>
      </div>
      <div style="font-family:Inter,sans-serif;font-size:11px;color:#34d399;
                  letter-spacing:3px;text-transform:uppercase;margin-bottom:20px;font-weight:500;">
        Psychometric Analysis · EFA · CFA · Synthetic Data
      </div>
      <h1 style="font-family:Inter,sans-serif;font-size:clamp(28px,4vw,52px);font-weight:800;
                 background:linear-gradient(135deg,#e2e8f0 40%,#a78bfa);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 line-height:1.1;margin:0 0 24px;letter-spacing:-1px;">
        Factor Analysis,<br>Built for Researchers
      </h1>
      <p style="font-family:Inter,sans-serif;font-size:17px;color:#94a3b8;
                max-width:560px;margin:0 auto 52px;line-height:1.8;">
        Run EFA and CFA on your survey data. Diagnose item issues, confirm
        factor structure, generate synthetic datasets — all in one rigorous pipeline.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
                max-width:860px;margin:0 auto 72px;">
      <div style="background:#111520;border:1px solid #1e2540;border-radius:14px;padding:28px 22px;text-align:center;">
        <div style="font-size:28px;margin-bottom:14px;">⬆</div>
        <div style="font-family:Inter,sans-serif;font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:8px;">Upload</div>
        <div style="font-family:Inter,sans-serif;font-size:13px;color:#64748b;line-height:1.7;">
          Upload your CSV or Excel survey dataset. Numeric columns detected automatically.
        </div>
      </div>
      <div style="background:#111520;border:1px solid #1e2540;border-top:2px solid #6c8dfa;
                  border-radius:14px;padding:28px 22px;text-align:center;">
        <div style="font-size:28px;margin-bottom:14px;">◈</div>
        <div style="font-family:Inter,sans-serif;font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:8px;">Analyse</div>
        <div style="font-family:Inter,sans-serif;font-size:13px;color:#64748b;line-height:1.7;">
          Run KMO, Bartlett, EFA and CFA with full diagnostics. Iterate freely — no limits.
        </div>
      </div>
      <div style="background:#111520;border:1px solid #1e2540;border-radius:14px;padding:28px 22px;text-align:center;">
        <div style="font-size:28px;margin-bottom:14px;">⬇</div>
        <div style="font-family:Inter,sans-serif;font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:8px;">Export</div>
        <div style="font-family:Inter,sans-serif;font-size:13px;color:#64748b;line-height:1.7;">
          Download cleaned data, synthetic datasets and a full Word analysis report (.docx).
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Free Trial CTA
    st.markdown("""
    <div style="max-width:860px;margin:0 auto 16px;padding:0 4px;">
      <div style="background:linear-gradient(135deg,rgba(108,141,250,0.07),rgba(167,139,250,0.06));
                  border:1px solid rgba(108,141,250,0.25);border-radius:18px;
                  padding:40px 36px;position:relative;overflow:hidden;text-align:center;">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,transparent,#6c8dfa 40%,#a78bfa 60%,transparent);"></div>
        <div style="font-family:Inter,sans-serif;font-size:10px;color:#34d399;
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;font-weight:600;">
          ◈ No credit card · Instant access
        </div>
        <div style="font-family:Inter,sans-serif;font-size:26px;font-weight:800;
                    background:linear-gradient(90deg,#e2e8f0,#6c8dfa);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    margin-bottom:12px;line-height:1.2;">Try EFActor Free</div>
        <p style="font-family:Inter,sans-serif;font-size:13px;color:#94a3b8;
                  max-width:500px;margin:0 auto 28px;line-height:1.9;">
          Upload data, run full EFA and CFA, generate synthetic datasets, preview all results.
          Export is unlocked with a paid access key.
        </p>
        <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;">
          <span style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:#34d399;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;">✓ Upload CSV / Excel</span>
          <span style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:#34d399;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;">✓ Unlimited EFA &amp; CFA runs</span>
          <span style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:#34d399;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;">✓ Full diagnostics &amp; plots</span>
          <span style="background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:#34d399;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;">✓ Synthetic data preview</span>
          <span style="background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.2);color:#f87171;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;opacity:.85;">✗ Data export (requires credits)</span>
          <span style="background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.2);color:#f87171;
                       font-family:Inter,sans-serif;font-size:11px;font-weight:500;padding:6px 14px;border-radius:8px;opacity:.85;">✗ Word report (.docx) requires credits</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, trial_col, _ = st.columns([1.4, 1, 1.4])
    with trial_col:
        trial_btn = st.button("◈ Start Free Trial", type="primary", use_container_width=True, key="trial_btn")
    st.markdown("<div style='margin-bottom:56px;'></div>", unsafe_allow_html=True)

    if trial_btn:
        st.session_state.update({
            "authenticated": True, "is_free_trial": True,
            "access_key": "FREE-TRIAL", "key_owner": "Free Trial", "credits": 0,
        })
        st.rerun()

    # Pricing header
    st.markdown("""
    <div style="text-align:center;margin-bottom:40px;">
      <div style="font-family:Inter,sans-serif;font-size:10px;color:#64748b;
                  letter-spacing:3px;text-transform:uppercase;margin-bottom:12px;font-weight:500;">Pricing</div>
      <div style="font-family:Inter,sans-serif;font-size:30px;font-weight:800;color:#e2e8f0;letter-spacing:-0.5px;">
        Simple, credit-based access
      </div>
      <div style="font-family:Inter,sans-serif;font-size:14px;color:#64748b;margin-top:10px;line-height:1.7;">
        Iterate freely. Credits are only used when you export.<br>
        Access key delivered to your email after purchase.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pricing cards
    p_cols = st.columns(3, gap="medium")
    for col, plan in zip(p_cols, PLANS):
        border_top = f"border-top:3px solid {plan['color']};" if plan["highlight"] else f"border-top:2px solid {plan['color']}44;"
        shadow = f"box-shadow:0 16px 48px {plan['color']}20;" if plan["highlight"] else ""
        badge_html = (
            '<div style="background:rgba(108,141,250,0.12);color:#6c8dfa;font-family:Inter,sans-serif;'
            'font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:4px 12px;'
            'border-radius:20px;display:inline-block;margin-bottom:16px;border:1px solid rgba(108,141,250,.3);">★ MOST POPULAR</div>'
            if plan["highlight"] else "<div style='height:28px;'></div>"
        )
        feats = "".join(
            f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#94a3b8;'
            f'padding:7px 0;border-bottom:1px solid #1a1f35;">'
            f'<span style="color:{plan["color"]};margin-right:8px;font-weight:600;">✓</span>{f}</div>'
            for f in plan["features"]
        )
        with col:
            st.markdown(f"""
            <div style="background:#0d1020;border:1px solid #1e2540;{border_top}
                        border-radius:16px;padding:32px 26px 16px;{shadow}
                        min-height:420px;box-sizing:border-box;">
              {badge_html}
              <div style="font-family:Inter,sans-serif;font-size:17px;font-weight:700;color:#e2e8f0;margin-bottom:6px;">{plan["name"]}</div>
              <div style="font-family:Inter,sans-serif;font-size:46px;font-weight:800;color:{plan["color"]};line-height:1;margin:10px 0 4px;">{plan["price"]}</div>
              <div style="font-family:Inter,sans-serif;font-size:11px;color:#64748b;margin-bottom:24px;font-weight:500;">{plan["credits"]} · {plan["period"]}</div>
              <div>{feats}</div>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(f"Get {plan['name']} →", url=plan["link"], use_container_width=True)

    # Credit cost table
    st.markdown("""
    <div style="max-width:860px;margin:56px auto 16px;">
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-family:Inter,sans-serif;font-size:16px;font-weight:700;color:#e2e8f0;margin-bottom:4px;">Export Credit Costs</div>
        <div style="font-family:Inter,sans-serif;font-size:12px;color:#64748b;">Analysis runs are always free. Credits only deducted on export.</div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
        <div style="background:#111520;border:1px solid #1e2540;border-radius:12px;padding:20px 16px;text-align:center;">
          <div style="font-size:22px;font-weight:800;color:#34d399;margin-bottom:4px;">1</div>
          <div style="font-size:10px;color:#64748b;font-weight:500;letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;">Credit</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.6;">Data export<br>≤ 300 rows</div>
        </div>
        <div style="background:#111520;border:1px solid #1e2540;border-radius:12px;padding:20px 16px;text-align:center;">
          <div style="font-size:22px;font-weight:800;color:#6c8dfa;margin-bottom:4px;">2</div>
          <div style="font-size:10px;color:#64748b;font-weight:500;letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;">Credits</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.6;">Data export<br>301 – 1,000 rows</div>
        </div>
        <div style="background:#111520;border:1px solid #1e2540;border-radius:12px;padding:20px 16px;text-align:center;">
          <div style="font-size:22px;font-weight:800;color:#a78bfa;margin-bottom:4px;">5</div>
          <div style="font-size:10px;color:#64748b;font-weight:500;letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;">Credits</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.6;">Data export<br>&gt; 1,000 rows</div>
        </div>
        <div style="background:#111520;border:1px solid #1e2540;border-radius:12px;padding:20px 16px;text-align:center;">
          <div style="font-size:22px;font-weight:800;color:#fbbf24;margin-bottom:4px;">1</div>
          <div style="font-size:10px;color:#64748b;font-weight:500;letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px;">Credit</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.6;">HTML analysis<br>report</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Access key entry
    st.markdown("""
    <div style="text-align:center;margin-top:72px;margin-bottom:24px;">
      <div style="font-family:Inter,sans-serif;font-size:13px;font-weight:600;
                  color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">
        Already have an access key?
      </div>
      <div style="font-family:Inter,sans-serif;font-size:12px;color:#374151;">
        Your key is emailed to you after purchase.
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([1.2, 1.6, 1.2])
    with c:
        key_input = st.text_input("Access Key", placeholder="EFA-XXXX-XXXX-XXXX",
                                   label_visibility="collapsed", key="landing_key_input")
        enter_btn = st.button("◈ Enter EFActor", type="primary", use_container_width=True)
        if enter_btn:
            if not key_input.strip():
                st.error("Please enter your access key.")
            else:
                with st.spinner("Validating key…"):
                    row = validate_key(key_input.strip())
                if row is None:
                    st.error("Invalid access key. Please check and try again.")
                elif int(row.get(COL_CREDITS, 0)) <= 0:
                    st.error("This key has 0 credits remaining. Please purchase a new plan.")
                else:
                    st.session_state.update({
                        "authenticated": True, "is_free_trial": False,
                        "access_key": key_input.strip(),
                        "key_owner": row.get(COL_OWNER, "Researcher"),
                        "credits": int(row.get(COL_CREDITS, 0)),
                    })
                    st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:40px;padding-bottom:16px;">
      <a href="https://your-purchase-link" target="_blank"
         style="font-family:Inter,sans-serif;font-size:12px;font-weight:600;color:white;
                background:linear-gradient(135deg,#151a2e,#1e2745);
                border:1px solid #1e2540;padding:10px 18px;border-radius:8px;
                margin:0 6px;display:inline-block;text-decoration:none;">👤 Get Access Key</a>
      <a href="mailto:your@email.com"
         style="font-family:Inter,sans-serif;font-size:12px;font-weight:600;color:white;
                background:linear-gradient(135deg,#151a2e,#1e2745);
                border:1px solid #1e2540;padding:10px 18px;border-radius:8px;
                margin:0 6px;display:inline-block;text-decoration:none;">⚙️ Support</a>
    </div>
    <div style="text-align:center;font-family:Inter,sans-serif;font-size:10px;
                color:#1e2540;margin-top:40px;letter-spacing:1px;padding-bottom:40px;">
      © 2025 EFActor · Psychometric Analysis Platform
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════
if not st.session_state.get("authenticated", False):
    render_landing()
    st.stop()

# Refresh live credit balance (paid users only)
if not st.session_state.get("is_free_trial", False):
    st.session_state["credits"] = get_credits(st.session_state["access_key"])


# ══════════════════════════════════════════════════════════════════
# EFA FUNCTIONS  (core logic unchanged)
# ══════════════════════════════════════════════════════════════════

def check_efa_suitability(df):
    kmo_all, kmo_model = calculate_kmo(df)
    chi2, p = calculate_bartlett_sphericity(df)
    labels = {.9:"Marvellous",.8:"Meritorious",.7:"Middling",.6:"Mediocre",.5:"Miserable"}
    kmo_label = next((v for k, v in sorted(labels.items(), reverse=True) if kmo_model >= k), "Unacceptable")
    return dict(kmo_model=round(float(kmo_model),4), kmo_label=kmo_label, kmo_pass=kmo_model>=0.6,
                bartlett_chi2=round(float(chi2),4), bartlett_p=round(float(p),6),
                bartlett_pass=p<0.05, overall_pass=kmo_model>=0.6 and p<0.05)

def determine_n_factors(df):
    max_factors = max(1, min(len(df.columns)-1, len(df)-1))
    fa = FactorAnalyzer(n_factors=max_factors, rotation=None)
    fa.fit(df)
    ev, _ = fa.get_eigenvalues()
    return dict(eigenvalues=ev.tolist(), suggested_n=max(1, int(np.sum(ev>1))))

def run_efa(df, n_factors, rotation="varimax"):
    n_factors = min(n_factors, len(df.columns)-1)
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df)
    factor_labels = [f"F{i+1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=df.columns, columns=factor_labels)
    communalities = pd.Series(fa.get_communalities(), index=df.columns, name="Communality")
    variance = pd.DataFrame(fa.get_factor_variance(),
                            index=["SS Loadings","Proportion Var","Cumulative Var"],
                            columns=factor_labels).T
    return dict(loadings=loadings, communalities=communalities, variance=variance, n_factors=n_factors)

def diagnose_loadings(loadings, communalities, load_thresh=0.4, comm_thresh=0.3):
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
        records.append(dict(Variable=var, MaxLoading=round(max_load,4),
                            FactorsAboveThreshold=n_high, Communality=round(comm,4),
                            Issue=", ".join(issues) if issues else "OK",
                            Severity=round(severity,4), RecommendDrop=len(issues)>0))
    return pd.DataFrame(records).sort_values("Severity", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# CFA FUNCTIONS  (core logic unchanged)
# ══════════════════════════════════════════════════════════════════

def build_cfa_model(loadings, threshold=0.4):
    factor_vars = {}
    for var in loadings.index:
        abs_row = np.abs(loadings.loc[var])
        if abs_row.max() >= threshold:
            best = abs_row.idxmax()
            factor_vars.setdefault(best, []).append(var)
    factor_vars = {f: v for f, v in factor_vars.items() if len(v) >= 2}
    model_str = "\n".join(f"{f} =~ " + " + ".join(v) for f, v in factor_vars.items())
    return model_str, factor_vars

def run_cfa(df, model_str):
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
        return dict(success=False, fit_indices={}, estimates=None, model_str=model_str, error=str(e))

def _parse_fit_indices(stats):
    flat = stats.iloc[0] if isinstance(stats, pd.DataFrame) else stats
    flat.index = [str(c).strip().upper() for c in flat.index]
    mapping = dict(CFI=["CFI"], TLI=["TLI","NNFI"], RMSEA=["RMSEA"], SRMR=["SRMR"],
                   Chi2=["CHI2","CHISQ","CHI-SQUARE","X2"], df=["DF","DOF"],
                   p_value=["P-VALUE","PVALUE","P_VALUE","P(CHI2)"], AIC=["AIC"], BIC=["BIC"])
    result = {}
    for label, candidates in mapping.items():
        for c in candidates:
            if c in flat.index:
                try: result[label] = round(float(flat[c]), 4)
                except: result[label] = flat[c]
                break
    return result

def assess_cfa_fit(fit_indices, thresholds):
    assessment = {}
    for idx, direction in [("CFI","≥"),("TLI","≥"),("RMSEA","≤"),("SRMR","≤")]:
        if idx in fit_indices and idx in thresholds:
            val, thresh = fit_indices[idx], thresholds[idx]
            passed = val >= thresh if direction == "≥" else val <= thresh
            assessment[idx] = dict(value=val, threshold=thresh, pass_=passed, direction=direction)
    n_pass = sum(1 for v in assessment.values() if v["pass_"])
    return dict(indices=assessment, n_pass=n_pass, n_total=len(assessment),
                overall_pass=(len(assessment)>0 and n_pass==len(assessment)))

def get_modification_suggestions(fit_assessment):
    suggestions, indices = [], fit_assessment.get("indices", {})
    checks = [
        ("RMSEA","RMSEA exceeds threshold. Consider freeing residual covariances between items sharing method variance, or removing items with high modification indices."),
        ("CFI",  "CFI is below threshold. Check whether indicators load on multiple factors. Consider adding cross-loadings or removing weak indicators."),
        ("SRMR", "SRMR exceeds threshold. Large residual correlations exist — review for systematic patterns among residuals."),
    ]
    for idx, msg in checks:
        if idx in indices and not indices[idx]["pass_"]:
            suggestions.append(f"{idx} = {indices[idx]['value']:.3f} — {msg}")
    if not suggestions and not fit_assessment.get("overall_pass"):
        suggestions.append("Model fit is inadequate. Consider revising factor structure, removing low-communality items, or allowing correlated residuals for items with common wording.")
    return suggestions


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA FUNCTIONS  (core logic unchanged)
# ══════════════════════════════════════════════════════════════════

def _make_psd(matrix, eps=1e-8):
    ev, evec = np.linalg.eigh(matrix)
    return evec @ np.diag(np.maximum(ev, eps)) @ evec.T

def generate_factor_based(df, efa_result, n_samples=500, seed=42):
    np.random.seed(seed)
    loadings = efa_result["loadings"].values
    communalities = efa_result["communalities"].values
    n_factors, columns = efa_result["n_factors"], efa_result["loadings"].index.tolist()
    factor_scores = np.random.multivariate_normal(np.zeros(n_factors), np.eye(n_factors), n_samples)
    common = factor_scores @ loadings.T
    unique = np.random.normal(0, np.sqrt(np.maximum(1-communalities, 1e-6)), (n_samples, len(columns)))
    synthetic = common + unique
    orig_mean, orig_std = df[columns].mean().values, df[columns].std().values
    orig_std = np.where(orig_std==0, 1.0, orig_std)
    syn_mean, syn_std = synthetic.mean(0), synthetic.std(0)
    syn_std = np.where(syn_std==0, 1.0, syn_std)
    rescaled = ((synthetic - syn_mean) / syn_std) * orig_std + orig_mean
    return pd.DataFrame(rescaled, columns=columns)

def generate_correlation_based(df, n_samples=500, seed=42):
    np.random.seed(seed)
    cov = _make_psd(df.cov().values)
    synthetic = np.random.multivariate_normal(df.mean().values, cov, n_samples)
    return pd.DataFrame(synthetic, columns=df.columns.tolist())

def validate_synthetic(original, synthetic):
    return pd.DataFrame([dict(
        Variable=col,
        OrigMean=round(original[col].mean(),4), SynMean=round(synthetic[col].mean(),4),
        OrigStd=round(original[col].std(),4),   SynStd=round(synthetic[col].std(),4),
        MeanDelta=round(abs(original[col].mean()-synthetic[col].mean()),4),
        StdDelta=round(abs(original[col].std()-synthetic[col].std()),4),
    ) for col in original.columns])


# ══════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS  (core logic unchanged)
# ══════════════════════════════════════════════════════════════════

def plot_scree(eigenvalues, suggested_n):
    x = list(range(1, len(eigenvalues)+1))
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

def plot_loading_heatmap(loadings, threshold=0.4):
    z = loadings.values
    variables, factors = loadings.index.tolist(), loadings.columns.tolist()
    annotations = [dict(x=f, y=v, text=f"{z[i][j]:.2f}", showarrow=False,
                        font=dict(color="white" if abs(z[i][j])>=threshold else C["muted"], size=11))
                   for i, v in enumerate(variables) for j, f in enumerate(factors)]
    fig = go.Figure(go.Heatmap(z=z, x=factors, y=variables, zmid=0, zmin=-1, zmax=1,
                               colorscale=[[0,"#1e1b4b"],[0.5,C["surface"]],[1,C["accent"]]],
                               colorbar=dict(title="Loading", tickfont=dict(color=C["text"]),
                                             title_font=dict(color=C["text"])),
                               hovertemplate="%{y} → %{x}<br>Loading: %{z:.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT_BASE, annotations=annotations,
                      title=dict(text="Factor Loading Heatmap", font=dict(color=C["accent2"])),
                      height=max(300, len(variables)*32+100), xaxis=dict(side="top"))
    return fig

def plot_communalities(communalities, comm_thresh=0.3):
    colors = [C["green"] if v>=0.5 else (C["yellow"] if v>=comm_thresh else C["red"])
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
                      yaxis=dict(title="Communality", gridcolor=C["border"], range=[0,1]))
    return fig

def plot_fit_indices(fit_assessment):
    indices = fit_assessment.get("indices", {})
    if not indices:
        return go.Figure()
    labels = list(indices.keys())
    values = [d["value"] for d in indices.values()]
    colors = [C["green"] if d["pass_"] else C["red"] for d in indices.values()]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[f"{v:.3f}" for v in values], textposition="outside",
                           hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    for label, data in indices.items():
        fig.add_annotation(x=label, y=data["threshold"],
                           text=f"Threshold: {data['threshold']}",
                           showarrow=True, arrowhead=2, arrowcolor=C["yellow"],
                           font=dict(color=C["yellow"], size=10), ay=-30)
    fig.update_layout(**LAYOUT_BASE, height=360, showlegend=False,
                      title=dict(text="CFA Fit Indices", font=dict(color=C["accent2"])),
                      xaxis=dict(title="Index", gridcolor=C["border"]),
                      yaxis=dict(title="Value", gridcolor=C["border"]))
    return fig

def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True).round(2)
    cols = corr.columns.tolist()
    annotations = [dict(x=c, y=r, text=f"{corr.loc[r,c]:.2f}", showarrow=False,
                        font=dict(size=9, color="white" if abs(corr.loc[r,c])>0.5 else C["text"]))
                   for r in corr.index for c in cols]
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="r", tickfont=dict(color=C["text"]),
                                             title_font=dict(color=C["text"])),
                               hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT_BASE, annotations=annotations,
                      title=dict(text="Correlation Matrix", font=dict(color=C["accent2"])),
                      height=max(350, len(cols)*30+120), xaxis=dict(tickangle=-35))
    return fig

def plot_synthetic_comparison(original, synthetic, max_vars=6):
    cols = original.columns[:max_vars].tolist()
    ncols = min(3, len(cols))
    nrows = (len(cols)+ncols-1)//ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols)
    for i, col in enumerate(cols):
        r, ci = i//ncols+1, i%ncols+1
        fig.add_trace(go.Histogram(x=original[col], name="Original", nbinsx=20,
                                   marker_color=C["accent"], opacity=0.6, showlegend=(i==0)), row=r, col=ci)
        fig.add_trace(go.Histogram(x=synthetic[col], name="Synthetic", nbinsx=20,
                                   marker_color=C["accent2"], opacity=0.6, showlegend=(i==0)), row=r, col=ci)
    fig.update_layout(**LAYOUT_BASE, barmode="overlay", height=300*nrows,
                      title=dict(text="Original vs Synthetic Distributions", font=dict(color=C["accent2"])),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(color=C["text"])))
    return fig


# ══════════════════════════════════════════════════════════════════
# DOCX REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_docx_report(original_df, cleaned_df, suitability, efa_result,
                         diagnostics, dropped_vars, cfa_result, fit_assessment,
                         cfa_thresholds, synthetic_validation=None, model_str=""):
    """Generate a Word (.docx) analysis report and return bytes."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = DocxDocument()

    # ── Page margins ────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1)
        section.right_margin  = Inches(1)

    # ── Helper styles ───────────────────────────────────────────────
    def _set_run_color(run, hex_color):
        run.font.color.rgb = RGBColor(
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
        )

    def _add_heading(text, level=1, color="#6c8dfa"):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after  = Pt(6)
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(16 if level == 1 else 13)
        _set_run_color(run, color)
        return p

    def _add_para(text, bold=False, color=None, size=10):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        if color:
            _set_run_color(run, color)
        return p

    def _shade_cell(cell, hex_color):
        """Apply background shading to a table cell."""
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color.lstrip("#"))
        tc_pr.append(shd)

    def _add_table(headers, rows_data, col_widths=None, header_bg="1e2540"):
        """Add a table with styled header row."""
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        # Header row
        hdr_cells = t.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            run = hdr_cells[i].paragraphs[0].runs[0]
            run.bold = True
            run.font.size = Pt(9)
            _set_run_color(run, "#6c8dfa")
            _shade_cell(hdr_cells[i], header_bg)
        # Data rows
        for row_vals, row_colors in rows_data:
            cells = t.add_row().cells
            for i, val in enumerate(row_vals):
                cells[i].text = str(val)
                run = cells[i].paragraphs[0].runs[0]
                run.font.size = Pt(9)
                if row_colors and row_colors[i]:
                    _set_run_color(run, row_colors[i])
        return t

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Title ────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title_p.add_run("◈ EFActor — Analysis Report")
    r.bold = True
    r.font.size = Pt(20)
    _set_run_color(r, "#6c8dfa")

    meta_p = doc.add_paragraph()
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta_p.add_run(f"Generated: {ts}")
    mr.font.size = Pt(9)
    _set_run_color(mr, "#94a3b8")

    doc.add_paragraph()

    # ── 1. Dataset Overview ──────────────────────────────────────────
    _add_heading("1. Dataset Overview", level=1)
    overview_rows = [
        (["Original Variables", str(len(original_df.columns))], [None, "#6c8dfa"]),
        (["After Cleaning",     str(len(cleaned_df.columns))],  [None, "#6c8dfa"]),
        (["Observations",       str(len(cleaned_df))],           [None, "#6c8dfa"]),
        (["Dropped Variables",  str(len(dropped_vars))],         [None, "#f87171" if dropped_vars else "#34d399"]),
    ]
    _add_table(["Metric", "Value"], overview_rows)
    if dropped_vars:
        _add_para(f"Dropped: {', '.join(dropped_vars)}", color="#f87171", size=9)

    # ── 2. EFA Suitability ───────────────────────────────────────────
    _add_heading("2. EFA Suitability", level=1)
    s = suitability
    suit_rows = [
        (["KMO Score",           f"{s['kmo_model']} — {s['kmo_label']}", "≥ 0.60",
          "PASS" if s["kmo_pass"] else "FAIL"],
         [None, None, None, "#34d399" if s["kmo_pass"] else "#f87171"]),
        (["Bartlett's Sphericity", f"χ² = {s['bartlett_chi2']}, p = {s['bartlett_p']}", "p < 0.05",
          "PASS" if s["bartlett_pass"] else "FAIL"],
         [None, None, None, "#34d399" if s["bartlett_pass"] else "#f87171"]),
        (["Overall",             "",  "",
          "PASS" if s["overall_pass"] else "FAIL"],
         [None, None, None, "#34d399" if s["overall_pass"] else "#f87171"]),
    ]
    _add_table(["Test", "Value", "Threshold", "Result"], suit_rows)

    # ── 3. EFA Results ───────────────────────────────────────────────
    _add_heading("3. Exploratory Factor Analysis", level=1)
    _add_para(f"Factors extracted: {efa_result['n_factors']}", bold=True, color="#6c8dfa")

    loadings      = efa_result["loadings"]
    communalities = efa_result["communalities"]
    variance      = efa_result["variance"]

    # Loadings table
    _add_heading("Factor Loadings", level=2, color="#a78bfa")
    load_headers = ["Variable"] + loadings.columns.tolist() + ["Communality"]
    load_rows = []
    for v in loadings.index:
        comm_val = communalities[v]
        comm_color = "#34d399" if comm_val >= 0.5 else ("#fbbf24" if comm_val >= 0.3 else "#f87171")
        vals   = [v] + [f"{loadings.loc[v, c]:.3f}" for c in loadings.columns] + [f"{comm_val:.3f}"]
        colors = [None] + [None] * len(loadings.columns) + [comm_color]
        load_rows.append((vals, colors))
    _add_table(load_headers, load_rows)

    # Variance table
    _add_heading("Variance Explained", level=2, color="#a78bfa")
    var_rows = [
        ([idx, f"{row['SS Loadings']:.4f}",
          f"{row['Proportion Var']*100:.2f}%",
          f"{row['Cumulative Var']*100:.2f}%"], [None]*4)
        for idx, row in variance.iterrows()
    ]
    _add_table(["Factor", "SS Loadings", "Proportion Var", "Cumulative Var"], var_rows)

    # Diagnostics table
    _add_heading("Item Diagnostics", level=2, color="#a78bfa")
    diag_rows = []
    for _, r in diagnostics.iterrows():
        comm = r["Communality"]
        comm_color = "#34d399" if comm >= 0.5 else ("#fbbf24" if comm >= 0.3 else "#f87171")
        issue      = r["Issue"]
        issue_color = "#34d399" if issue == "OK" else ("#fbbf24" if "Cross" in issue else "#f87171")
        drop_color  = "#f87171" if r["RecommendDrop"] else "#34d399"
        drop_text   = "✗ Drop" if r["RecommendDrop"] else "✓ Keep"
        diag_rows.append((
            [r["Variable"], str(r["MaxLoading"]), str(r["FactorsAboveThreshold"]),
             str(comm), issue, drop_text],
            [None, None, None, comm_color, issue_color, drop_color]
        ))
    _add_table(["Variable", "Max Load", "# Factors ≥ Threshold", "Communality", "Issue", "Recommended"],
               diag_rows)

    # ── 4. CFA ──────────────────────────────────────────────────────
    if cfa_result and cfa_result.get("success") and fit_assessment:
        _add_heading("4. Confirmatory Factor Analysis", level=1)
        fa = fit_assessment
        overall_text  = "ADEQUATE" if fa["overall_pass"] else "INADEQUATE"
        overall_color = "#34d399"  if fa["overall_pass"] else "#f87171"
        _add_para(f"Model Fit: {overall_text}  ({fa['n_pass']}/{fa['n_total']} indices passed)",
                  bold=True, color=overall_color)
        fit_rows = [
            ([idx, str(d["value"]), f"{d['direction']} {d['threshold']}",
              "PASS" if d["pass_"] else "FAIL"],
             [None, None, None, "#34d399" if d["pass_"] else "#f87171"])
            for idx, d in fa["indices"].items()
        ]
        _add_table(["Index", "Value", "Threshold", "Result"], fit_rows)

        _add_heading("Model Specification", level=2, color="#a78bfa")
        doc.add_paragraph(model_str or "—")

    # ── 5. Synthetic Data Validation ────────────────────────────────
    if synthetic_validation is not None:
        _add_heading("5. Synthetic Data Validation", level=1)
        syn_rows = [
            ([r["Variable"], str(r["OrigMean"]), str(r["SynMean"]),
              str(r["OrigStd"]),  str(r["SynStd"]),
              str(r["MeanDelta"]), str(r["StdDelta"])], [None]*7)
            for _, r in synthetic_validation.iterrows()
        ]
        _add_table(["Variable", "Orig Mean", "Syn Mean", "Orig Std",
                    "Syn Std", "Mean Δ", "Std Δ"], syn_rows)

    # ── Footer ───────────────────────────────────────────────────────
    doc.add_paragraph()
    foot_p = doc.add_paragraph()
    foot_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = foot_p.add_run(
        "EFActor — Results should be interpreted in context of your research design "
        "and theoretical framework."
    )
    fr.font.size = Pt(8)
    _set_run_color(fr, "#64748b")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = dict(df_original=None, df_working=None, suitability=None,
                 n_factors_auto=None, eigenvalues=None, efa_result=None,
                 diagnostics=None, efa_done=False, cfa_result=None,
                 fit_assessment=None, synthetic_factor=None, synthetic_corr=None,
                 syn_validation=None, report_docx=None)
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
if "dropped_vars" not in st.session_state:
    st.session_state["dropped_vars"] = []
S = st.session_state

is_trial   = S.get("is_free_trial", False)
credits    = S.get("credits", 0)
key_owner  = S.get("key_owner", "Researcher")
access_key = S.get("access_key", "")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    cred_color   = "#34d399" if (is_trial or credits > 10) else ("#fbbf24" if credits > 3 else "#f87171")
    cred_display = "Trial" if is_trial else str(credits)
    cred_label   = "Free Trial — export locked" if is_trial else "Credits remaining"
    cred_bar_w   = "100" if is_trial else str(min(100, int(credits * 2)))

    st.markdown(f"""
    <div style="padding:16px 0 20px;border-bottom:1px solid #1e2540;margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#6c8dfa,#a78bfa);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:16px;flex-shrink:0;">◈</div>
        <div>
          <div style="font-family:Inter,sans-serif;font-size:16px;font-weight:800;
                      background:linear-gradient(90deg,#6c8dfa,#a78bfa);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">EFActor</div>
          <div style="font-family:Inter,sans-serif;font-size:9px;color:#64748b;
                      letter-spacing:2px;text-transform:uppercase;font-weight:500;">Psychometric Analysis</div>
        </div>
      </div>
      <div style="background:#0a0c14;border:1px solid #1e2540;border-radius:10px;padding:12px 14px;">
        <div style="font-family:Inter,sans-serif;font-size:9px;color:#64748b;
                    letter-spacing:2px;text-transform:uppercase;font-weight:500;margin-bottom:6px;">Account</div>
        <div style="font-family:Inter,sans-serif;font-size:13px;font-weight:600;
                    color:#e2e8f0;margin-bottom:8px;">{key_owner}</div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <div style="font-family:Inter,sans-serif;font-size:10px;color:#64748b;">{cred_label}</div>
          <div style="font-family:Inter,sans-serif;font-size:18px;font-weight:800;
                      color:{cred_color};">{cred_display}</div>
        </div>
        <div style="background:#1e2540;border-radius:4px;height:3px;overflow:hidden;">
          <div style="background:{cred_color};height:3px;width:{cred_bar_w}%;border-radius:4px;"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sign Out button
    if st.button("↩ Sign Out", use_container_width=True, key="signout_btn"):
        for k in ["authenticated","access_key","key_owner","credits","is_free_trial",
                  "_last_file_key","df_original","df_working","suitability","n_factors_auto",
                  "eigenvalues","efa_result","diagnostics","efa_done","dropped_vars",
                  "cfa_result","fit_assessment","synthetic_factor","synthetic_corr",
                  "syn_validation","report_docx"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)

    st.markdown("### ⚙️ EFA Settings")
    loading_threshold    = st.slider("Loading threshold",    0.30, 0.60, 0.40, 0.05)
    communality_threshold= st.slider("Communality threshold",0.20, 0.60, 0.30, 0.05)
    rotation_method      = st.selectbox("Rotation method",
                             ["varimax","oblimin","promax","quartimax","equamax"])

    st.divider()
    st.markdown("### 📐 CFA Thresholds")
    cfi_thresh   = st.slider("CFI ≥",   0.80, 0.99, 0.95, 0.01)
    tli_thresh   = st.slider("TLI ≥",   0.80, 0.99, 0.95, 0.01)
    rmsea_thresh = st.slider("RMSEA ≤", 0.04, 0.15, 0.06, 0.01)
    srmr_thresh  = st.slider("SRMR ≤",  0.04, 0.15, 0.08, 0.01)
    cfa_thresholds = dict(CFI=cfi_thresh, TLI=tli_thresh, RMSEA=rmsea_thresh, SRMR=srmr_thresh)

    st.divider()
    st.markdown("### 🧪 Synthetic Data")
    syn_n    = st.number_input("Sample size",  50, 10000, 500, 50)
    syn_seed = st.number_input("Random seed",   0,  9999,  42)

    st.divider()
    if st.button("🔄 Reset Analysis", use_container_width=True):
        for k in ["_last_file_key","df_original","df_working","suitability","n_factors_auto",
                  "eigenvalues","efa_result","diagnostics","efa_done","dropped_vars",
                  "cfa_result","fit_assessment","synthetic_factor","synthetic_corr",
                  "syn_validation","report_docx"]:
            st.session_state.pop(k, None)
        st.rerun()

    if is_trial:
        st.divider()
        st.markdown("""
        <div style="background:rgba(108,141,250,.07);border:1px solid rgba(108,141,250,.2);
                    border-radius:10px;padding:14px 16px;text-align:center;">
          <div style="font-family:Inter,sans-serif;font-size:12px;font-weight:700;
                      color:#6c8dfa;margin-bottom:6px;">Unlock Exports</div>
          <div style="font-family:Inter,sans-serif;font-size:11px;color:#64748b;line-height:1.6;margin-bottom:10px;">
            Get credits to download datasets and reports.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("Get Access Key →", "https://your-payment-link", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
  <div style="width:40px;height:40px;background:linear-gradient(135deg,#6c8dfa,#a78bfa);
              border-radius:10px;display:flex;align-items:center;justify-content:center;
              font-size:20px;flex-shrink:0;">◈</div>
  <div>
    <div style="font-family:Inter,sans-serif;font-size:28px;font-weight:800;
                background:linear-gradient(90deg,#6c8dfa,#a78bfa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:-0.5px;line-height:1;">EFActor</div>
    <div style="font-family:Inter,sans-serif;font-size:11px;color:#64748b;
                letter-spacing:2px;text-transform:uppercase;font-weight:500;margin-top:2px;">
      Psychometric Analysis Platform
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("_Upload your dataset and run a complete EFA → CFA pipeline. Diagnose items, confirm structure, generate synthetic data, and export your results._")

if is_trial:
    st.markdown('<div class="warn-box">⚡ <b>Free Trial mode</b> — all analysis runs are unlimited. Data export requires an access key with credits. <a href="https://your-payment-link" style="color:#fbbf24;">Get access →</a></div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">1</span> Upload Dataset</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns([2, 1])
with col_up1:
    uploaded = st.file_uploader("Upload CSV or Excel",
                                 type=["csv","CSV","xlsx","xls","XLSX","XLS"],
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
                st.error("❌ Unsupported file type.")
                st.stop()
            # Fix: coerce string-encoded values, drop all-NaN cols, then drop NaN rows
            df_coerced = df_raw.apply(lambda col: pd.to_numeric(col, errors="coerce"))
            df_numeric = (df_coerced
                          .select_dtypes(include=[np.number])
                          .dropna(axis=1, how="all")
                          .dropna())
            if len(df_numeric.columns) < 3:
                st.error("❌ Need at least 3 numeric variables. Non-numeric columns are excluded automatically.")
                st.stop()
            if len(df_numeric) < 30:
                st.warning("⚠️ Fewer than 30 observations — results may be unreliable.")
            S.df_original = df_numeric.copy()
            S.df_working  = df_numeric.copy()
            S.dropped_vars = []
            for k in ["suitability","efa_result","cfa_result","fit_assessment","efa_done",
                      "synthetic_factor","synthetic_corr","syn_validation","report_docx",
                      "n_factors_auto","eigenvalues","diagnostics"]:
                S[k] = None if k != "efa_done" else False
            S["_last_file_key"] = _file_key
        except Exception as e:
            st.error(f"❌ Could not parse file: {e}")
            st.stop()

if S.df_original is None:
    st.markdown('<div class="warn-box">👆 Upload a dataset to begin.</div>', unsafe_allow_html=True)
    st.stop()

with st.expander("📋 Dataset Preview", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [(c1,len(S.df_original),"Observations"),(c2,len(S.df_original.columns),"Variables"),
                             (c3,len(S.df_working.columns),"Working Vars"),(c4,len(S.dropped_vars),"Dropped")]:
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
    log_event(access_key, "suitability_run",
              f"kmo={'pass' if S.suitability['kmo_pass'] else 'fail'},bartlett={'pass' if S.suitability['bartlett_pass'] else 'fail'},suggested_factors={S.n_factors_auto},vars={len(S.df_working.columns)},rows={len(S.df_working)}")

if S.suitability:
    suit = S.suitability
    pill = "pill-pass" if suit["overall_pass"] else "pill-fail"
    text = "✓ SUITABLE FOR EFA" if suit["overall_pass"] else "✗ EFA SUITABILITY ISSUES"
    st.markdown(f'<b>Overall:</b> <span class="{pill}">{text}</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{suit["kmo_model"]}</div>'
                f'<div class="metric-label">KMO Score — {suit["kmo_label"]}</div>'
                f'<div style="margin-top:8px"><span class="{"pill-pass" if suit["kmo_pass"] else "pill-fail"}">{"PASS" if suit["kmo_pass"] else "FAIL"}</span></div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{suit["bartlett_p"]}</div>'
                f'<div class="metric-label">Bartlett p-value (χ²={suit["bartlett_chi2"]})</div>'
                f'<div style="margin-top:8px"><span class="{"pill-pass" if suit["bartlett_pass"] else "pill-fail"}">{"PASS" if suit["bartlett_pass"] else "FAIL"}</span></div></div>',
                unsafe_allow_html=True)
    if not suit["overall_pass"]:
        st.markdown('<div class="warn-box">⚠️ Suitability concerns detected. You can still proceed — interpret results cautiously.</div>', unsafe_allow_html=True)
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
    _max_factors = max(2, len(S.df_working.columns) - 1)
    default_n = S.n_factors_auto if S.n_factors_auto else 2
    default_n = int(min(max(1, default_n), _max_factors))  # clamp to valid range
    n_factors = st.number_input("Number of factors to extract", 1, _max_factors, default_n, 1)
    if st.button("▶ Run EFA", use_container_width=True):
        with st.spinner("Running factor analysis…"):
            S.efa_result  = run_efa(S.df_working, n_factors, rotation_method)
            S.diagnostics = diagnose_loadings(S.efa_result["loadings"],
                                              S.efa_result["communalities"],
                                              loading_threshold, communality_threshold)
            S.efa_done = True
            _efa_event = "efa_rerun" if S.get("_efa_ever_run") else "efa_first_run"
            S["_efa_ever_run"] = True
            log_event(access_key, _efa_event, f"n_factors={n_factors},rotation={rotation_method},vars={len(S.df_working.columns)}")
    if S.efa_result:
        t1, t2, t3, t4 = st.tabs(["Factor Loadings","Variance Explained","Loading Heatmap","Communalities"])
        with t1: st.dataframe(S.efa_result["loadings"].round(4), use_container_width=True)
        with t2: st.dataframe(S.efa_result["variance"].round(4), use_container_width=True)
        with t3: st.plotly_chart(plot_loading_heatmap(S.efa_result["loadings"], loading_threshold), use_container_width=True)
        with t4: st.plotly_chart(plot_communalities(S.efa_result["communalities"], communality_threshold), use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 4 — DIAGNOSE & DROP
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">4</span> Diagnose &amp; Refine Items</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    # ── Colour-coded diagnostics table ──────────────────────────────
    def _diag_table_html(df):
        rows_html = ""
        for _, r in df.iterrows():
            # Communality colouring
            comm = r["Communality"]
            if comm >= 0.5:
                comm_style = "color:#34d399;font-weight:600;"
                comm_icon  = "🟢"
            elif comm >= 0.3:
                comm_style = "color:#fbbf24;font-weight:600;"
                comm_icon  = "🟡"
            else:
                comm_style = "color:#f87171;font-weight:600;"
                comm_icon  = "🔴"

            # Issue colouring
            issue = r["Issue"]
            if issue == "OK":
                issue_style = "color:#34d399;font-weight:600;"
                issue_icon  = "✓ OK"
            elif "Cross-Loader" in issue:
                issue_style = "color:#fbbf24;font-weight:600;"
                issue_icon  = f"⚠ {issue}"
            else:
                issue_style = "color:#f87171;font-weight:600;"
                issue_icon  = f"✗ {issue}"

            # RecommendDrop colouring
            drop = r["RecommendDrop"]
            if drop:
                drop_style = "color:#f87171;font-weight:700;"
                drop_cell  = "✗ Drop"
            else:
                drop_style = "color:#34d399;font-weight:600;"
                drop_cell  = "✓ Keep"

            rows_html += (
                f"<tr>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;'>{r['Variable']}</td>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;'>{r['MaxLoading']}</td>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;text-align:center;'>{r['FactorsAboveThreshold']}</td>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;{comm_style}'>{comm_icon} {comm}</td>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;{issue_style}'>{issue_icon}</td>"
                f"<td style='padding:7px 10px;border-bottom:1px solid #1e2540;{drop_style}'>{drop_cell}</td>"
                f"</tr>"
            )
        return f"""
<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:.85rem;font-family:Inter,sans-serif;">
<thead>
<tr style="background:#1e2540;">
  <th style="padding:8px 10px;text-align:left;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;">Variable</th>
  <th style="padding:8px 10px;text-align:left;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;">Max Loading</th>
  <th style="padding:8px 10px;text-align:center;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;"># Factors ≥ Threshold</th>
  <th style="padding:8px 10px;text-align:left;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;">Communality</th>
  <th style="padding:8px 10px;text-align:left;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;">Issue</th>
  <th style="padding:8px 10px;text-align:left;color:#6c8dfa;font-size:.75rem;text-transform:uppercase;letter-spacing:1px;">Recommended</th>
</tr>
</thead>
<tbody style="color:#e2e8f0;">
{rows_html}
</tbody>
</table>
</div>
"""
    st.markdown(_diag_table_html(S.diagnostics), unsafe_allow_html=True)
    problem_vars = S.diagnostics[S.diagnostics["RecommendDrop"]]["Variable"].tolist()
    if problem_vars:
        st.markdown(f'<div class="warn-box">⚠️ <b>{len(problem_vars)}</b> variable(s) flagged: {", ".join(problem_vars)}</div>', unsafe_allow_html=True)

    available = S.df_working.columns.tolist()
    to_drop = st.multiselect("Select variables to drop (optional)", available,
                              default=[v for v in problem_vars if v in available])
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("▶ Apply Drops & Re-run EFA", use_container_width=True):
            if to_drop:
                S.df_working   = S.df_working.drop(columns=to_drop)
                S.dropped_vars = list(set(S.dropped_vars + to_drop))
            with st.spinner("Re-running EFA…"):
                S.efa_result  = run_efa(S.df_working, n_factors, rotation_method)
                S.diagnostics = diagnose_loadings(S.efa_result["loadings"],
                                                  S.efa_result["communalities"],
                                                  loading_threshold, communality_threshold)
            log_event(access_key, "efa_rerun_after_drop", f"dropped={len(to_drop)},remaining_vars={len(S.df_working.columns)}")
            st.success(f"✓ EFA re-run. Working set: {len(S.df_working.columns)} variables.")
    with col_d2:
        if st.button("↩ Restore All Variables", use_container_width=True):
            S.df_working   = S.df_original.copy()
            S.dropped_vars = []
            S.efa_result   = None
            S.diagnostics  = None
            S.efa_done     = False
            st.rerun()
    if S.dropped_vars:
        st.markdown(f'<div class="info-box">📋 Dropped so far: <b>{", ".join(S.dropped_vars)}</b></div>', unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 5 — CFA
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">5</span> Confirmatory Factor Analysis (CFA)</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) first.</div>', unsafe_allow_html=True)
else:
    model_str, factor_vars = build_cfa_model(S.efa_result["loadings"], loading_threshold)
    with st.expander("📝 Auto-generated CFA Model", expanded=False):
        edited_model = st.text_area("Edit model specification if needed:", value=model_str, height=160)
    if st.button("▶ Run CFA", use_container_width=True):
        if not edited_model.strip():
            st.error("Model specification is empty.")
        else:
            with st.spinner("Fitting CFA model…"):
                S.cfa_result = run_cfa(S.df_working, edited_model)
                if S.cfa_result["success"]:
                    S.fit_assessment = assess_cfa_fit(S.cfa_result["fit_indices"], cfa_thresholds)
                    S.cfa_result["model_str"] = edited_model
                    _fit = S.fit_assessment
                    log_event(access_key, "cfa_run_success",
                              f"overall={'pass' if _fit['overall_pass'] else 'fail'},indices_passed={_fit['n_pass']}/{_fit['n_total']}")
                else:
                    log_event(access_key, "cfa_run_failed", f"error={str(S.cfa_result.get('error',''))[:80]}")
    if S.cfa_result:
        if not S.cfa_result["success"]:
            st.error(f"CFA failed: {S.cfa_result['error']}")
        else:
            fa = S.fit_assessment
            pill2 = "pill-pass" if fa["overall_pass"] else "pill-fail"
            text2 = "✓ MODEL FIT ADEQUATE" if fa["overall_pass"] else "✗ MODEL FIT INADEQUATE"
            st.markdown(f'<b>Fit:</b> <span class="{pill2}">{text2}</span> — {fa["n_pass"]}/{fa["n_total"]} indices passed', unsafe_allow_html=True)
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
            log_event(access_key, "synthetic_generated", f"method=factor_based,n={syn_n}")
            with st.spinner("Simulating from factor structure…"):
                S.synthetic_factor = generate_factor_based(S.df_working, S.efa_result, n_samples=syn_n, seed=syn_seed)
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_factor)
            st.success(f"✓ {syn_n} synthetic observations generated.")
    with c2:
        st.markdown("#### Correlation Preserving")
        st.markdown('<div class="info-box">Multivariate normal from empirical <b>covariance matrix</b>. Faster, preserves pairwise correlations but not latent structure explicitly.</div>', unsafe_allow_html=True)
        if st.button("▶ Generate (Correlation-Based)", use_container_width=True):
            log_event(access_key, "synthetic_generated", f"method=correlation_based,n={syn_n}")
            with st.spinner("Sampling from multivariate normal…"):
                S.synthetic_corr = generate_correlation_based(S.df_working, n_samples=syn_n, seed=syn_seed)
                S.syn_validation = validate_synthetic(S.df_working, S.synthetic_corr)
            st.success(f"✓ {syn_n} synthetic observations generated.")

    syn_display = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr
    if syn_display is not None:
        t1, t2, t3 = st.tabs(["Preview","Validation Summary","Distribution Comparison"])
        with t1: st.dataframe(syn_display.head(10).round(3), use_container_width=True)
        with t2:
            if S.syn_validation is not None:
                st.dataframe(S.syn_validation, use_container_width=True, hide_index=True)
        with t3: st.plotly_chart(plot_synthetic_comparison(S.df_working, syn_display), use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# STEP 7 — EXPORT  (credit-gated)
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge"><span class="step-num">7</span> Export Results</div>', unsafe_allow_html=True)

if S.efa_result is None:
    st.markdown('<div class="warn-box">Complete EFA (Step 3) to enable exports.</div>', unsafe_allow_html=True)
else:
    syn_export = S.synthetic_factor if S.synthetic_factor is not None else S.synthetic_corr
    syn_label  = "factor_based" if S.synthetic_factor is not None else "correlation_based"

    # ── Trial: show lock ─────────────────────────────────────────
    if is_trial:
        st.markdown("""
        <div class="lock-box">
          <div style="font-family:Inter,sans-serif;font-size:15px;font-weight:700;
                      color:#6c8dfa;margin-bottom:8px;">◈ Export is a Paid Feature</div>
          <div style="font-family:Inter,sans-serif;font-size:13px;color:#94a3b8;line-height:1.7;margin-bottom:12px;">
            You're on the Free Trial — all analysis above is fully functional and unlimited.<br>
            To download datasets and reports, purchase an access key with credits.<br><br>
            <b style="color:#e2e8f0;">Credit costs:</b>&nbsp;
            ≤ 300 rows = 1 credit &nbsp;·&nbsp;
            301–1,000 rows = 2 credits &nbsp;·&nbsp;
            &gt; 1,000 rows = 5 credits &nbsp;·&nbsp;
            Word report (.docx) = 1 credit
          </div>
        </div>
        """, unsafe_allow_html=True)
        log_event(access_key, "export_wall_hit", "trial_user")
        _, upg_col, _ = st.columns([1, 2, 1])
        with upg_col:
            st.link_button("◈ Get Access Key →", "https://your-payment-link", use_container_width=True)
        st.stop()

    # ── Paid: show exports ───────────────────────────────────────
    current_credits = st.session_state.get("credits", 0)
    REPORT_COST = 1

    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        st.markdown("##### 🗃️ Cleaned Dataset")
        n_clean = len(S.df_working)
        cost_clean = export_credit_cost(n_clean)
        st.markdown(f'<div class="info-box" style="font-size:.82rem;">{n_clean} rows × {len(S.df_working.columns)} cols<br><b>Cost: {cost_clean} credit{"s" if cost_clean>1 else ""}</b></div>', unsafe_allow_html=True)
        if current_credits < cost_clean:
            log_event(access_key, "export_blocked", f"type=cleaned_data,needed={cost_clean},have={current_credits}")
            st.markdown(f'<div class="warn-box" style="font-size:.82rem;">⚠️ Need {cost_clean} credits ({current_credits} remaining).</div>', unsafe_allow_html=True)
            st.link_button("Top up →", "https://your-payment-link", use_container_width=True)
        else:
            if st.button(f"⬇ Download Cleaned Data ({cost_clean} cr)", use_container_width=True, key="dl_clean_btn"):
                log_event(access_key, "export_attempt", f"type=cleaned_data,rows={n_clean},cost={cost_clean}")
                new_bal = deduct_credits(access_key, cost_clean)
                st.session_state["credits"] = new_bal
                log_event(access_key, "export_complete", f"type=cleaned_data,rows={n_clean},credits_remaining={new_bal}")
                st.download_button("⬇ cleaned_data.csv",
                                   data=S.df_working.to_csv(index=False).encode("utf-8"),
                                   file_name="efactor_cleaned_data.csv", mime="text/csv",
                                   use_container_width=True, key="dl_clean_actual")
                st.success(f"✓ {cost_clean} credit(s) deducted. Balance: {new_bal}")

    with col_e2:
        st.markdown("##### 🧪 Synthetic Dataset")
        if syn_export is not None:
            n_syn = len(syn_export)
            cost_syn = export_credit_cost(n_syn)
            st.markdown(f'<div class="info-box" style="font-size:.82rem;">{n_syn} rows × {len(syn_export.columns)} cols<br><b>Cost: {cost_syn} credit{"s" if cost_syn>1 else ""}</b></div>', unsafe_allow_html=True)
            if current_credits < cost_syn:
                st.markdown(f'<div class="warn-box" style="font-size:.82rem;">⚠️ Need {cost_syn} credits ({current_credits} remaining).</div>', unsafe_allow_html=True)
                st.link_button("Top up →", "https://your-payment-link", use_container_width=True)
            else:
                if st.button(f"⬇ Download Synthetic Data ({cost_syn} cr)", use_container_width=True, key="dl_syn_btn"):
                    log_event(access_key, "export_attempt", f"type=synthetic_data,rows={n_syn},cost={cost_syn}")
                    new_bal = deduct_credits(access_key, cost_syn)
                    st.session_state["credits"] = new_bal
                    log_event(access_key, "export_complete", f"type=synthetic_data,rows={n_syn},credits_remaining={new_bal}")
                    st.download_button(f"⬇ synthetic_{syn_label}.csv",
                                       data=syn_export.to_csv(index=False).encode("utf-8"),
                                       file_name=f"efactor_synthetic_{syn_label}.csv", mime="text/csv",
                                       use_container_width=True, key="dl_syn_actual")
                    st.success(f"✓ {cost_syn} credit(s) deducted. Balance: {new_bal}")
        else:
            st.markdown('<div class="warn-box">Generate synthetic data in Step 6 first.</div>', unsafe_allow_html=True)

    with col_e3:
        st.markdown("##### 📄 Analysis Report")
        has_cfa = S.cfa_result is not None and S.cfa_result["success"]
        if has_cfa:
            st.markdown(f'<div class="info-box" style="font-size:.82rem;">Full EFA + CFA Word report (.docx)<br><b>Cost: {REPORT_COST} credit</b></div>', unsafe_allow_html=True)
            if current_credits < REPORT_COST:
                st.markdown(f'<div class="warn-box" style="font-size:.82rem;">⚠️ Insufficient credits.</div>', unsafe_allow_html=True)
                st.link_button("Top up →", "https://your-payment-link", use_container_width=True)
            else:
                if st.button(f"🔨 Build Word Report ({REPORT_COST} cr)", use_container_width=True, key="build_rpt"):
                    log_event(access_key, "export_attempt", f"type=word_report,cost={REPORT_COST}")
                    with st.spinner("Compiling Word report…"):
                        S.report_docx = generate_docx_report(
                            original_df=S.df_original, cleaned_df=S.df_working,
                            suitability=S.suitability, efa_result=S.efa_result,
                            diagnostics=S.diagnostics, dropped_vars=S.dropped_vars,
                            cfa_result=S.cfa_result, fit_assessment=S.fit_assessment,
                            cfa_thresholds=cfa_thresholds, synthetic_validation=S.syn_validation,
                            model_str=S.cfa_result.get("model_str",""))
                    new_bal = deduct_credits(access_key, REPORT_COST)
                    st.session_state["credits"] = new_bal
                    log_event(access_key, "export_complete", f"type=word_report,credits_remaining={new_bal}")
                    st.success(f"✓ Report ready. {REPORT_COST} credit deducted. Balance: {new_bal}")
                if S.report_docx:
                    st.download_button("⬇ efa_cfa_report.docx",
                                       data=S.report_docx,
                                       file_name="efactor_report.docx",
                                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       use_container_width=True)
        else:
            st.markdown('<div class="warn-box">Run CFA (Step 5) to enable report generation.</div>', unsafe_allow_html=True)

    # ── ZIP bundle ────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 📦 Full Export Bundle (.zip)")
    zip_rows = max(len(S.df_working), len(syn_export) if syn_export is not None else 0)
    zip_cost = export_credit_cost(zip_rows) + (REPORT_COST if S.report_docx else 0)
    st.markdown(f'<div class="info-box" style="font-size:.82rem;">Cleaned data + synthetic data + Word report + loadings CSV + model spec<br><b>Cost: {zip_cost} credits</b></div>', unsafe_allow_html=True)
    if current_credits < zip_cost:
        st.markdown(f'<div class="warn-box" style="font-size:.82rem;">⚠️ Need {zip_cost} credits ({current_credits} remaining).</div>', unsafe_allow_html=True)
        st.link_button("Top up credits →", "https://your-payment-link", use_container_width=True)
    else:
        if st.button(f"⬇ Build & Download ZIP ({zip_cost} cr)", use_container_width=True, key="dl_zip_btn"):
            log_event(access_key, "export_attempt", f"type=zip_bundle,cost={zip_cost}")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("cleaned_data.csv", S.df_working.to_csv(index=False))
                if syn_export is not None:
                    zf.writestr(f"synthetic_{syn_label}.csv", syn_export.to_csv(index=False))
                if S.report_docx:
                    zf.writestr("efa_cfa_report.docx", S.report_docx)
                if S.cfa_result and S.cfa_result.get("model_str"):
                    zf.writestr("cfa_model.txt", S.cfa_result["model_str"])
                if S.efa_result:
                    zf.writestr("efa_loadings.csv", S.efa_result["loadings"].round(4).to_csv())
                    zf.writestr("efa_communalities.csv", S.efa_result["communalities"].round(4).to_frame().to_csv())
            zip_buffer.seek(0)
            new_bal = deduct_credits(access_key, zip_cost)
            st.session_state["credits"] = new_bal
            ts_str = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button("⬇ efactor_bundle.zip",
                               data=zip_buffer.getvalue(),
                               file_name=f"efactor_bundle_{ts_str}.zip",
                               mime="application/zip", key="dl_zip_actual")
            log_event(access_key, "export_complete", f"type=zip_bundle,credits_remaining={new_bal}")
            st.success(f"✓ {zip_cost} credits deducted. Balance: {new_bal}")


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:8px 0 24px;">
  <div style="font-family:Inter,sans-serif;font-size:14px;font-weight:700;
               background:linear-gradient(90deg,#6c8dfa,#a78bfa);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:8px;">◈ EFActor</div>
  <div style="font-family:Inter,sans-serif;font-size:11px;color:#374151;">
    Psychometric Analysis Platform &nbsp;·&nbsp; Built for researchers.
    Interpret results within your theoretical framework and research design.
  </div>
  <div style="font-family:Inter,sans-serif;font-size:10px;color:#1e2540;margin-top:8px;">
    © 2025 EFActor
  </div>
</div>
""", unsafe_allow_html=True)
