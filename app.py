import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import io 
import pickle  

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_OK = True
except ImportError:
    LIGHTGBM_OK = False

def download_chart(fig, key):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label="Download Chart",
        data=buf,
        file_name="chart.png",
        mime="image/png",
        key=key
    )

@st.cache_data
def compute_outliers(df, num_cols):
    _df  = df.copy()
    rows = []
    for col in num_cols:
        if col not in _df.columns:
            continue
        s = _df[col].dropna()
        if s.empty:
            continue
        q1, q3       = s.quantile(0.25), s.quantile(0.75)
        iqr          = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count        = int(((s < lower) | (s > upper)).sum())
        rows.append({
            "Column"       : col,
            "Lower Bound"  : round(float(lower), 3),
            "Upper Bound"  : round(float(upper), 3),
            "Outlier Count": count,
            "Outlier %"    : round((count / len(s)) * 100, 2) if len(s) else 0,
        })
    return rows

@st.cache_data
def load_data(file_bytes):
    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError("Uploaded CSV file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), engine="pyarrow")
        if df.empty:
            raise ValueError("CSV has no data rows.")
        if df.shape[1] == 0:
            raise ValueError("CSV has no columns.")
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
            )
        return df
    except Exception as e:
        raise ValueError(f"Invalid CSV file: {str(e)}")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="DataPilot Studio", layout="wide", page_icon="🛸")

# ================= STYLING =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #c9d1d9;
}

#MainMenu, footer, .stDeployButton { display: none; }

/* ── Hero ── */
.hero {
    padding: 2.6rem 0 1.8rem 0;
    margin-bottom: 0;
    position: relative;
}

.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #4a5260;
    margin-bottom: 0.7rem;
}

.hero-title {
    font-size: 2.9rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    color: #f0f6fc;
    margin-bottom: 0.6rem;
}

.hero-title .dot {
    color: #f78166;
}

.hero-title .dim {
    color: #2478ff;
    font-weight: 600;
}

.hero-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.06em;
    margin-bottom: 1.3rem;
}

.hero-desc span {
    color: #388bfd;
}

/* ── Signature scan-line ── */
.scan-line {
    position: relative;
    height: 2px;
    width: 100%;
    margin-bottom: 2rem;
    background: #161b22;
    border-radius: 2px;
    overflow: hidden;
}

.scan-line::after {
    content: '';
    position: absolute;
    top: 0; left: -30%;
    height: 2px;
    width: 30%;
    background: linear-gradient(to right, transparent, #388bfd, #f78166, transparent);
}

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
    transition: border-color 0.15s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #388bfd !important;
}

/* ── Banners ── */
.banner {
    border-radius: 8px;
    padding: 0.8rem 1.1rem;
    margin: 0.7rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.banner-ok   { background: #0d1f17; border: 1px solid #1a4032; border-left: 3px solid #3fb950; color: #3fb950; }
.banner-warn { background: #1a1300; border: 1px solid #3d2e00; border-left: 3px solid #d29922; color: #d29922; }
.banner-err  { background: #1a0d0d; border: 1px solid #3d1515; border-left: 3px solid #f85149; color: #f85149; }
.banner-info { background: #0d1620; border: 1px solid #17324a; border-left: 3px solid #388bfd; color: #79c0ff; }

/* ── Section headers ── */
.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 1.6rem 0 0.15rem 0;
    letter-spacing: -0.01em;
}

.section-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    color: #4a5260;
    letter-spacing: 0.04em;
    margin-bottom: 0.7rem;
}

.section-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #8b949e;
    margin-bottom: 0.8rem;
}

.soft-divider {
    height: 1px;
    background: linear-gradient(to right, #21262d, transparent);
    margin: 1.3rem 0 1.3rem 0;
}

/* ── Stat cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 1.2rem 0;
}

.stat-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 26px; height: 26px;
    background: radial-gradient(circle at top left, var(--accent, #388bfd) 0%, transparent 72%);
    opacity: 0.5;
}

.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #388bfd);
    opacity: 0.85;
}

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #8b949e;
    margin-bottom: 0.45rem;
    position: relative;
}

.stat-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1;
    color: #f0f6fc;
    position: relative;
}

.stat-value.blue   { color: #388bfd; --accent: #388bfd; }
.stat-value.green  { color: #3fb950; --accent: #3fb950; }
.stat-value.yellow { color: #d29922; --accent: #d29922; }
.stat-value.red    { color: #f85149; --accent: #f85149; }

/* ── Pill-style segmented controls ── */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
    border-radius: 20px !important;
    border: 1px solid #21262d !important;
    padding: 0.35rem 1rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="secondary"] {
    background: #0d1117 !important;
    color: #8b949e !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: #388bfd !important;
    color: #f0f6fc !important;
}

.stButton > button[kind="primary"] {
    background: #161b22 !important;
    color: #388bfd !important;
    border-color: #388bfd !important;
}

/* ── Tabs (top-level: EDA / Model) ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #8b949e;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    padding: 0.45rem 0.95rem;
    transition: color 0.15s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #c9d1d9;
}

.stTabs [aria-selected="true"] {
    background: #161b22 !important;
    color: #388bfd !important;
}

/* Nested (sub) tabs — slightly smaller so the hierarchy reads clearly
   against the top-level EDA / Model tabs above them */
.stTabs .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border: none;
    padding: 0;
    margin-top: 0.4rem;
}

.stTabs .stTabs [data-baseweb="tab"] {
    font-size: 0.63rem;
    padding: 0.38rem 0.8rem;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Subheaders (native st.markdown ###) ── */
h3 {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #8b949e !important;
    letter-spacing: 0.01em;
    margin-top: 1.5rem !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ================= SHARED UI HELPERS =================
def sec(title, subtitle=None):
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub_html}', unsafe_allow_html=True)

def eyebrow(title):
    st.markdown(f'<p class="section-eyebrow">{title}</p>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

def banner(kind, text):
    icons = {"ok": "✓", "warn": "⚠", "err": "✕", "info": "ℹ"}
    st.markdown(f'<div class="banner banner-{kind}">{icons.get(kind, "•")} &nbsp;{text}</div>', unsafe_allow_html=True)


# ================= HERO =================
st.markdown("""
<div class="hero">
    <div class="hero-title">DataPilot <span class="dim">Studio</span></div>
    <div class="hero-desc">
        Upload CSV <span>→</span> Explore &amp; Clean <span>→</span> Train Models
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

# ================= FILE UPLOAD =================
file = st.file_uploader(
    "Upload CSV file (Max size: 25MB)",
    type=["csv"],
    label_visibility="collapsed"
)

if file is not None:
    if file.size > 25 * 1024 * 1024:
        banner("err", "File too large — upload under <strong>25MB</strong>.")
        st.stop()

    if "file_name" not in st.session_state or st.session_state.file_name != file.name:
        try:
            st.session_state.file_name   = file.name
            st.session_state.original_df = load_data(file.read())
            st.session_state.df          = st.session_state.original_df.copy()
        except ValueError as e:
            banner("err", str(e))
            st.stop()

    if "df" not in st.session_state:
        banner("warn", "Data not initialized — please re-upload.")
        st.stop()

    df = st.session_state.df

    size_mb = file.size / (1024 * 1024)
    banner("ok", f"<strong>{file.name}</strong> loaded &nbsp;·&nbsp; {size_mb:.2f} MB")

    if len(df) > 200000:
        banner("warn", "Large dataset — some operations may be slow.")

    # ── Stat cards ──
    missing_total = int(df.isnull().sum().sum())
    dup_total     = int(df.duplicated().sum())
    miss_cls      = "yellow" if missing_total > 0 else "green"
    dup_cls       = "yellow" if dup_total > 0 else "green"

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card" style="--accent:#388bfd">
            <div class="stat-label">Rows</div>
            <div class="stat-value blue">{df.shape[0]:,}</div>
        </div>
        <div class="stat-card" style="--accent:#388bfd">
            <div class="stat-label">Columns</div>
            <div class="stat-value blue">{df.shape[1]}</div>
        </div>
        <div class="stat-card" style="--accent:{'#d29922' if missing_total > 0 else '#3fb950'}">
            <div class="stat-label">Missing Values</div>
            <div class="stat-value {miss_cls}">{missing_total:,}</div>
        </div>
        <div class="stat-card" style="--accent:{'#d29922' if dup_total > 0 else '#3fb950'}">
            <div class="stat-label">Duplicate Rows</div>
            <div class="stat-value {dup_cls}">{dup_total:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ================= TOP-LEVEL TABS: EDA / MODEL =================
    main_tab_eda, main_tab_model = st.tabs(["🔭 EDA Lab", "🤖 Model Lab"])


    # ---------------- EDA LAB ----------------
    with main_tab_eda:
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5, eda_tab6 = st.tabs([
            "Overview", "Column Analyzer", "Correlation",
            "Data Cleaning", "Duplicate Rows", "Visualization"
        ])

        # ---------- OVERVIEW ----------
        with eda_tab1:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            object_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
            bool_cols    = df.select_dtypes(include=["bool"]).columns.tolist()

            outlier_rows = compute_outliers(df, tuple(numeric_cols)) if numeric_cols else []

            # ── 1. Dataset Preview ──
            eyebrow("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            divider()

            # ── 2. Column Info ──
            eyebrow("Column Info")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p style="font-size:0.78rem;color:#8b949e;margin-bottom:0.3rem;">Column Names</p>', unsafe_allow_html=True)
                st.write(df.columns.tolist())
            with c2:
                st.markdown('<p style="font-size:0.78rem;color:#8b949e;margin-bottom:0.3rem;">Data Types</p>', unsafe_allow_html=True)
                st.write(df.dtypes)
            divider()

            # ── 3. Statistical Summary ──
            eyebrow("Statistical Summary")
            if numeric_cols:
                st.markdown('<p style="font-size:0.75rem;color:#388bfd;margin-bottom:0.3rem;">▸ Numerical Columns</p>', unsafe_allow_html=True)
                st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
            if object_cols:
                st.markdown('<p style="font-size:0.75rem;color:#3fb950;margin-bottom:0.3rem;">▸ Categorical Columns</p>', unsafe_allow_html=True)
                st.dataframe(df[object_cols].describe(), use_container_width=True)
            if bool_cols:
                st.markdown('<p style="font-size:0.75rem;color:#d29922;margin-bottom:0.3rem;">▸ Boolean Columns</p>', unsafe_allow_html=True)
                st.dataframe(df[bool_cols].describe(), use_container_width=True)
            divider()

            # ── 4. Missing Values ──
            eyebrow("Missing Values")
            missing          = df.isnull().sum()
            missing_pct_col  = (missing / len(df)) * 100 if len(df) > 0 else missing * 0
            missing_df       = pd.DataFrame({
                "Missing Values": missing,
                "Percentage (%)": missing_pct_col.round(2)
            })
            missing_filtered = missing_df[missing_df["Missing Values"] > 0]
            if missing_filtered.empty:
                banner("ok", "No missing values found in this dataset.")
            else:
                st.dataframe(missing_filtered, use_container_width=True)
            divider()

            # ── 5. Data Health Score ──
            eyebrow("Data Health Score")

            total_cells   = df.shape[0] * df.shape[1]
            missing_pct   = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
            dup_pct       = (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0
            outlier_total = sum(r["Outlier Count"] for r in outlier_rows)
            outlier_pct   = min(100, (outlier_total / len(df) * 100)) if len(df) > 0 else 0

            missing_score = max(0, 100 - missing_pct * 2)
            dup_score     = max(0, 100 - dup_pct * 3)
            outlier_score = max(0, 100 - outlier_pct * 1.5)
            health_score  = round(missing_score * 0.4 + dup_score * 0.3 + outlier_score * 0.3, 1)

            if health_score >= 80:
                score_color, score_label, score_icon = "#3fb950", "Excellent", "✓"
            elif health_score >= 60:
                score_color, score_label, score_icon = "#d29922", "Fair", "⚠"
            else:
                score_color, score_label, score_icon = "#f85149", "Poor", "✕"

            st.markdown(f"""
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:12px;
                        padding:1.5rem 1.7rem;margin-bottom:1rem;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;right:0;width:60px;height:60px;
                            background:radial-gradient(circle at top right, {score_color} 0%, transparent 70%);
                            opacity:0.25;"></div>
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#8b949e;
                            text-transform:uppercase;letter-spacing:0.16em;margin-bottom:0.6rem;position:relative;">
                    Overall Health Score
                </div>
                <div style="display:flex;align-items:baseline;gap:0.6rem;position:relative;">
                    <div style="font-size:2.9rem;font-weight:800;color:{score_color};line-height:1;">
                        {health_score}<span style="font-size:1rem;color:#4a5260;">/100</span>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:{score_color};">
                        {score_icon} {score_label}
                    </div>
                </div>
                <div style="margin-top:1rem;background:#161b22;border-radius:6px;height:6px;overflow:hidden;position:relative;">
                    <div style="width:{health_score}%;height:100%;background:{score_color};
                                border-radius:6px;transition:width 0.6s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Missing Score",   f"{round(missing_score, 1)}/100", f"{missing_pct:.1f}% missing", delta_color="inverse")
            sc2.metric("Duplicate Score", f"{round(dup_score, 1)}/100", f"{dup_pct:.1f}% duplicates", delta_color="inverse")
            sc3.metric("Outlier Score",   f"{round(outlier_score, 1)}/100", f"{outlier_pct:.1f}% outliers", delta_color="inverse")

            divider()

            # ── 6. Outlier Detection Table ──
            eyebrow("Outlier Detection (IQR Method)")
            if not numeric_cols:
                banner("warn", "No numeric columns found for outlier detection.")
            else:
                outlier_df = (
                    pd.DataFrame(outlier_rows)
                    .sort_values("Outlier Count", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(outlier_df, use_container_width=True)

        # ---------- COLUMN ANALYZER ----------
        with eda_tab2:
            df = st.session_state.df

            if len(df.columns) == 0:
                banner("warn", "No columns remaining — go to **Data Cleaning** tab and reset the dataset.")
            else:
                column = st.selectbox("Select Column", df.columns, key="col_analyzer")

                divider()

                col_series   = df[column]
                missing_cnt  = int(col_series.isnull().sum())
                missing_pct  = (missing_cnt / len(df) * 100) if len(df) > 0 else 0

                colA, colB, colC, colD = st.columns(4)
                colA.metric("Data Type",      str(col_series.dtype))
                colB.metric("Missing Values", missing_cnt)
                colC.metric("Missing %",      f"{missing_pct:.2f}%")
                colD.metric("Unique Values",  int(col_series.nunique()))

                divider()

                clean_all = col_series.dropna()

                if clean_all.empty:
                    banner("warn", f"<strong>{column}</strong> has no non-null values -- nothing to analyze.")

                # ── NUMERIC ──
                elif pd.api.types.is_numeric_dtype(col_series):

                    clean_all = clean_all.astype(float)
                    mean_val   = float(clean_all.mean())
                    median_val = float(clean_all.median())
                    std_val    = float(clean_all.std()) if len(clean_all) > 1 else 0.0
                    min_val    = float(clean_all.min())
                    max_val    = float(clean_all.max())
                    is_constant = clean_all.nunique() <= 1

                    skew_val = float(clean_all.skew()) if not is_constant else 0.0
                    kurt_val = float(clean_all.kurt()) if not is_constant else 0.0
                    skew_val = 0.0 if pd.isna(skew_val) else skew_val
                    kurt_val = 0.0 if pd.isna(kurt_val) else kurt_val

                    colE, colF, colG = st.columns(3)
                    colE.metric("Mean",    round(mean_val, 2))
                    colF.metric("Median",  round(median_val, 2))
                    colG.metric("Std Dev", round(std_val, 2))

                    colH, colI, colJ = st.columns(3)
                    colH.metric("Min",      round(min_val, 2))
                    colI.metric("Max",      round(max_val, 2))
                    colJ.metric("Skewness", round(skew_val, 2))

                    divider()

                    # ── Skewness & Kurtosis Interpretation ──
                    eyebrow("Distribution Interpretation")

                    if is_constant:
                        banner("info", "This column has only one distinct value -- skewness and kurtosis aren't meaningful here.")
                    else:
                        if skew_val > 1:
                            skew_label, skew_color = "Highly Right Skewed", "#f85149"
                        elif skew_val > 0.5:
                            skew_label, skew_color = "Moderately Right Skewed", "#d29922"
                        elif skew_val < -1:
                            skew_label, skew_color = "Highly Left Skewed", "#f85149"
                        elif skew_val < -0.5:
                            skew_label, skew_color = "Moderately Left Skewed", "#d29922"
                        else:
                            skew_label, skew_color = "Approximately Symmetric", "#3fb950"

                        if kurt_val > 3:
                            kurt_label, kurt_color = "Leptokurtic -- heavy tails, sharp peak (outliers likely)", "#f85149"
                        elif kurt_val < -1:
                            kurt_label, kurt_color = "Platykurtic -- light tails, flat distribution", "#d29922"
                        else:
                            kurt_label, kurt_color = "Mesokurtic -- normal-like tails", "#3fb950"

                        interp_l, interp_r = st.columns(2)
                        interp_l.markdown(f"""
                        <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid {skew_color};
                                    border-radius:8px;padding:0.8rem 1rem;">
                            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#8b949e;
                                        text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem;">
                                Skewness · {round(skew_val, 3)}
                            </div>
                            <div style="font-size:0.78rem;color:{skew_color};">{skew_label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        interp_r.markdown(f"""
                        <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid {kurt_color};
                                    border-radius:8px;padding:0.8rem 1rem;">
                            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#8b949e;
                                        text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem;">
                                Kurtosis · {round(kurt_val, 3)}
                            </div>
                            <div style="font-size:0.78rem;color:{kurt_color};">{kurt_label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    divider()

                    # ── Charts ──
                    clean = clean_all
                    if len(clean) > 5000:
                        clean = clean.sample(5000, random_state=42)

                    can_kde = clean.nunique() >= 2

                    chart_l, chart_r = st.columns(2)

                    with chart_l:
                        eyebrow("Distribution")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        fig.patch.set_facecolor("#0d1117")
                        ax.set_facecolor("#0d1117")
                        ax.hist(clean, bins=30, color="#388bfd", alpha=0.75, edgecolor="none")
                        if can_kde:
                            ax2 = ax.twinx()
                            clean.plot.kde(ax=ax2, color="#f78166", linewidth=1.5)
                            ax2.set_ylabel("")
                            ax2.tick_params(left=False, right=False, labelleft=False, labelright=False)
                            ax2.set_facecolor("#0d1117")
                            for s in ax2.spines.values(): s.set_visible(False)
                        ax.axvline(mean_val, color="#d29922", linewidth=1.2, linestyle="--", label="Mean")
                        ax.axvline(median_val, color="#3fb950", linewidth=1.2, linestyle="--", label="Median")
                        ax.set_xlabel(column, color="#8b949e", fontsize=8)
                        ax.set_ylabel("Count", color="#8b949e", fontsize=8)
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        for s in ax.spines.values(): s.set_visible(False)
                        ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117", edgecolor="#21262d")
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, key=f"hist_download_{column}")
                        plt.close(fig)

                    with chart_r:
                        eyebrow("Box Plot")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        fig.patch.set_facecolor("#0d1117")
                        ax.set_facecolor("#0d1117")
                        ax.boxplot(clean, vert=False, patch_artist=True, widths=0.5,
                                boxprops=dict(facecolor="#161b22", color="#388bfd"),
                                medianprops=dict(color="#f78166", linewidth=2),
                                whiskerprops=dict(color="#388bfd"),
                                capprops=dict(color="#388bfd"),
                                flierprops=dict(marker="o", color="#d29922", markersize=3, alpha=0.5))
                        ax.set_xlabel(column, color="#8b949e", fontsize=8)
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        ax.set_yticks([])
                        for s in ax.spines.values(): s.set_visible(False)
                        ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, key=f"box_download_{column}")
                        plt.close(fig)

                    divider()

                    # ── Outlier Detection (reuses the shared helper) ──
                    eyebrow("Outlier Detection (IQR Method)")

                    col_outlier_info = compute_outliers(df, (column,))
                    if not col_outlier_info:
                        banner("info", "Outlier bounds couldn't be computed for this column.")
                    else:
                        info      = col_outlier_info[0]
                        lower_b   = info["Lower Bound"]
                        upper_b   = info["Upper Bound"]
                        out_count = info["Outlier Count"]

                        outlier_mask = (col_series.astype(float) < lower_b) | (col_series.astype(float) > upper_b)
                        outlier_rows_df = df[outlier_mask]

                        oc1, oc2, oc3 = st.columns(3)
                        oc1.metric("Lower Bound", lower_b)
                        oc2.metric("Upper Bound", upper_b)
                        oc3.metric("Outlier Rows", out_count)

                        if outlier_rows_df.empty:
                            banner("ok", "No outliers found in this column.")
                        else:
                            with st.expander(f"Preview {min(50, len(outlier_rows_df))} Outlier Rows"):
                                st.dataframe(outlier_rows_df.head(50), use_container_width=True)
                                if len(outlier_rows_df) > 50:
                                    st.caption(f"Showing 50 of {len(outlier_rows_df):,} outlier rows.")

                # ── CATEGORICAL ──
                else:
                    vc = col_series.value_counts()

                    mode_val = col_series.mode()
                    if not mode_val.empty:
                        most_frequent = mode_val.iloc[0]
                        top_count     = vc.iloc[0]
                    else:
                        most_frequent = "N/A"
                        top_count     = 0

                    colE, colF = st.columns(2)
                    colE.metric("Most Frequent Value", str(most_frequent))
                    colF.metric("Top Value Count", int(top_count))

                    divider()

                    # ── Value Distribution ──
                    eyebrow("Value Distribution")
                    vc_table = pd.DataFrame({
                        "Value": vc.index,
                        "Count": vc.values,
                        "Percentage": (vc.values / len(df) * 100).round(2) if len(df) > 0 else 0
                    })
                    vc_table["Percentage"] = vc_table["Percentage"].astype(str) + " %"
                    st.dataframe(vc_table, use_container_width=True)

                    divider()

                    MAX_BARS  = 20
                    vc_plot   = vc.head(MAX_BARS)
                    truncated = len(vc) > MAX_BARS

                    chart_l, chart_r = st.columns(2)

                    with chart_l:
                        eyebrow(f"Top {MAX_BARS} Values" if truncated else "Value Counts")
                        fig, ax = plt.subplots(figsize=(5, max(3, len(vc_plot) * 0.35)))
                        fig.patch.set_facecolor("#0d1117")
                        ax.set_facecolor("#0d1117")
                        colors = ["#388bfd" if i == 0 else "#161b22" for i in range(len(vc_plot))]
                        ax.barh(vc_plot.index.astype(str)[::-1], vc_plot.values[::-1],
                                color=colors[::-1], edgecolor="none")
                        ax.set_xlabel("Count", color="#8b949e", fontsize=8)
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        for s in ax.spines.values(): s.set_visible(False)
                        ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, key=f"bar_download_{column}")
                        plt.close(fig)

                    with chart_r:
                        if len(vc) <= 10:
                            eyebrow("Distribution")
                            pie_data = vc
                        else:
                            eyebrow("Top 10 Share")
                            top10    = vc.head(10)
                            other    = vc.iloc[10:].sum()
                            pie_data = pd.concat([top10, pd.Series({"Other": other})])

                        pie_colors = ["#388bfd", "#58a6ff", "#3fb950", "#d29922",
                                    "#f78166", "#bc8cff", "#79c0ff", "#56d364",
                                    "#e3b341", "#ff7b72", "#8b949e"]

                        fig, ax = plt.subplots(figsize=(4, 4))
                        fig.patch.set_facecolor("#0d1117")
                        wedges, texts, autotexts = ax.pie(
                            pie_data.values,
                            labels=pie_data.index.astype(str),
                            colors=pie_colors[:len(pie_data)],
                            autopct="%1.1f%%", startangle=140,
                            textprops={"color": "#8b949e", "fontsize": 7},
                            wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
                        )
                        for at in autotexts:
                            at.set_color("#f0f6fc")
                            at.set_fontsize(7)
                        ax.set_facecolor("#0d1117")
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, key=f"pie_download_{column}")
                        plt.close(fig)

        # ---------- CORRELATION ----------
        with eda_tab3:
            all_numeric_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
            ]

            if len(all_numeric_cols) < 2:
                banner("warn", "Not enough numeric or boolean columns for correlation -- need at least 2.")
            else:
                eyebrow("Correlation Setup")

                target_col = st.selectbox(
                    "Target column",
                    options=all_numeric_cols,
                    key="corr_target"
                )

                other_options = [c for c in all_numeric_cols if c != target_col]

                sp1, sp2 = st.columns([4, 1])
                with sp1:
                    selected_others = st.multiselect(
                        "Compare against (numeric / boolean columns)",
                        options=other_options,
                        key="corr_selected_others"
                    )
                with sp2:
                    st.markdown('<div style="height:1.55rem;"></div>', unsafe_allow_html=True)

                    def _corr_select_all_cb():
                        st.session_state.corr_selected_others = other_options[:19]

                    st.button(
                        "Select all", key="corr_select_all", use_container_width=True,
                        on_click=_corr_select_all_cb,
                        disabled=len(other_options) == 0
                    )

                if len(other_options) > 19:
                    banner(
                        "info",
                        f"{len(other_options)} candidate columns available -- \"Select all\" picks the first 19 "
                        f"(20 total with the target) to keep the heatmap readable. Deselect / reselect manually for a different set."
                    )

                if len(selected_others) > 19:
                    banner("err", f"Too many columns selected ({len(selected_others) + 1} with target). Please keep it to 20 total -- deselect {len(selected_others) - 19} column(s).")
                elif not selected_others:
                    banner("warn", "Select at least one column to compare against the target.")
                else:
                    divider()

                    corr_cols = [target_col] + selected_others
                    corr_df   = df[corr_cols].apply(pd.to_numeric, errors="coerce")
                    corr      = corr_df.corr()
                    num_cols  = len(corr_cols)

                    st.markdown("""
                    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                                padding:0.8rem 1.1rem;margin-bottom:1rem;
                                font-family:'Space Mono',monospace;font-size:0.72rem;color:#8b949e;">
                        <span style="color:#f0f6fc;font-weight:700;">Correlation Guide &nbsp;·&nbsp;</span>
                        <span style="color:#3fb950;">+1 = Perfect Positive</span> &nbsp;·&nbsp;
                        <span style="color:#d29922;">0 = No Relation</span> &nbsp;·&nbsp;
                        <span style="color:#f85149;">-1 = Perfect Negative</span>
                        <br><span style="font-size:0.65rem;color:#8b949e;margin-top:0.3rem;display:block;">
                            |0.8–1.0| Strong &nbsp;·&nbsp; |0.5–0.8| Moderate &nbsp;·&nbsp; |0.2–0.5| Weak &nbsp;·&nbsp; |0.0–0.2| Very Weak
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Table: target's correlation with each selected column ──
                    eyebrow(f"Correlation with \"{target_col}\"")

                    target_corr = corr[target_col].drop(target_col)

                    def get_strength(val):
                        if pd.isna(val):
                            return "N/A"
                        abs_val   = abs(val)
                        direction = "Positive" if val > 0 else "Negative"
                        if abs_val >= 0.8:
                            return f"Strong {direction}"
                        elif abs_val >= 0.5:
                            return f"Moderate {direction}"
                        elif abs_val >= 0.2:
                            return f"Weak {direction}"
                        else:
                            return "Very Weak"

                    target_table = pd.DataFrame({
                        "Column": target_corr.index,
                        "Correlation": target_corr.values.round(4),
                    })
                    target_table["Strength"] = target_table["Correlation"].apply(get_strength)
                    target_table = target_table.sort_values("Correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

                    def color_corr(val):
                        if pd.isna(val):
                            return "background-color:#0d1117;color:#4a5260;"

                        abs_val = abs(val)
                        if val > 0:
                            if abs_val >= 0.8:
                                bg, fg = "#0c2340", "#60a5fa"
                            elif abs_val >= 0.5:
                                bg, fg = "#0a1a2e", "#79c0ff"
                            elif abs_val >= 0.2:
                                bg, fg = "#0d1826", "#5b9ee8"
                            else:
                                bg, fg = "#0d1117", "#4a7fb5"
                        elif val < 0:
                            if abs_val >= 0.8:
                                bg, fg = "#3a0d16", "#ff6b7a"
                            elif abs_val >= 0.5:
                                bg, fg = "#2b0b12", "#f78166"
                            elif abs_val >= 0.2:
                                bg, fg = "#1d0d12", "#e56b6f"
                            else:
                                bg, fg = "#0d1117", "#a85b63"
                        else:
                            bg, fg = "#0d1117", "#4a5260"

                        return f"background-color:{bg};color:{fg};font-weight:500;"

                    styled_target_table = target_table.style.map(color_corr, subset=["Correlation"])
                    st.dataframe(styled_target_table, use_container_width=True, hide_index=True)

                    divider()

                    # ── Full matrix table ──
                    eyebrow("Full Correlation Matrix")
                    styled_matrix = corr.round(3).style.map(color_corr)
                    st.dataframe(styled_matrix, use_container_width=True)

                    divider()

                    # ── Heatmap diagram ──
                    eyebrow("Correlation Heatmap")
                    annot = num_cols <= 15

                    from matplotlib.colors import LinearSegmentedColormap
                    horizon_cmap = LinearSegmentedColormap.from_list(
                        "horizon_corr",
                        ["#0b1f3a", "#174ea6", "#111827", "#2563eb", "#7dd3fc"],
                        N=256
                    )

                    fig4, ax4 = plt.subplots(figsize=(max(6, num_cols * 0.6), max(5, num_cols * 0.5)))
                    fig4.patch.set_facecolor("#0d1117")
                    ax4.set_facecolor("#0d1117")
                    sns.heatmap(corr, annot=False, cmap=horizon_cmap, center=0, vmin=-1, vmax=1, ax=ax4,
                                linewidths=0.7, linecolor="#0d1117", square=True, cbar_kws={"shrink": 0.80, "pad": 0.03})

                    if annot:
                        for i in range(corr.shape[0]):
                            for j in range(corr.shape[1]):
                                value = corr.iloc[i, j]
                                text_color = "#ffffff" if abs(value) >= 0.55 else "#c7d0d9"
                                ax4.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center",
                                        fontsize=8, fontweight="bold", color=text_color)

                    ax4.tick_params(colors="#8b949e", labelsize=7, length=0)
                    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
                    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)

                    target_idx = list(corr.columns).index(target_col)
                    target_color = "#60a5fa"
                    ax4.add_patch(plt.Rectangle((target_idx, 0), 1, num_cols, fill=False, edgecolor=target_color, linewidth=2.2))
                    ax4.add_patch(plt.Rectangle((0, target_idx), num_cols, 1, fill=False, edgecolor=target_color, linewidth=2.2))

                    cbar = ax4.collections[0].colorbar
                    cbar.ax.tick_params(colors="#8b949e", labelsize=7, length=0)
                    cbar.outline.set_edgecolor("#263241")
                    cbar.outline.set_linewidth(0.8)

                    for spine in ax4.spines.values():
                        spine.set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig4)
                    download_chart(fig4, "corr_heatmap_download")
                    plt.close(fig4)

        # ---------- DATA CLEANING ----------
        with eda_tab4:
            df = st.session_state.df

            def push_cleaning_history():
                history = st.session_state.setdefault("cleaning_history", [])
                history.append(st.session_state.df.copy())
                if len(history) > 20:
                    history.pop(0)

            # ── Missing Values Table ──
            eyebrow("Missing Values Table")

            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if missing.empty:
                banner("ok", "No missing values in this dataset.")
            else:
                missing_df = pd.DataFrame({
                    "Column": missing.index,
                    "Missing Values": missing.values
                }).sort_values(by="Missing Values", ascending=False).reset_index(drop=True)

                st.caption("Only columns with missing values are shown below")
                st.dataframe(missing_df, use_container_width=True)

                divider()

                # ── Fill Missing Values ──
                eyebrow("Fill Missing Values")

                col1, col2 = st.columns(2)
                with col1:
                    selected_col = st.selectbox(
                        "Select Column",
                        missing_df["Column"].tolist(),
                        key="col_select"
                    )

                is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
                method_options = ["Mean", "Median", "Mode", "Custom Value"] if is_numeric else ["Mode", "Custom Value"]

                with col2:
                    method = st.selectbox("Select Method", method_options, key="method_select")

                custom_value = None
                if method == "Custom Value":
                    custom_value = st.text_input(
                        "Enter custom value to fill",
                        placeholder='e.g. 0 or "Unknown"',
                        key="custom_fill_val"
                    )

                if st.button("Fill Missing", key="fill_btn"):
                    value = None
                    if method == "Mean":
                        value = df[selected_col].mean()
                    elif method == "Median":
                        value = df[selected_col].median()
                    elif method == "Mode":
                        mode_val = df[selected_col].mode()
                        if mode_val.empty:
                            banner("warn", f"<strong>{selected_col}</strong> has no non-null values to compute a mode from.")
                        else:
                            value = mode_val.iloc[0]
                    else:
                        if not custom_value:
                            banner("warn", "Please enter a custom value.")
                        elif is_numeric:
                            try:
                                value = float(custom_value)
                            except ValueError:
                                banner("err", "This is a numeric column -- please enter a number.")
                        else:
                            value = custom_value

                    if value is not None:
                        push_cleaning_history()
                        df[selected_col] = df[selected_col].fillna(value)
                        st.session_state.df = df
                        st.session_state["last_action"] = f"'{selected_col}' filled using {method}"
                        st.rerun()

            # ── Success message ──
            if "last_action" in st.session_state:
                banner("ok", st.session_state["last_action"])
                del st.session_state["last_action"]

            divider()

            # ── Add Column (From Ranges) ──
            eyebrow("Add Column (From Ranges)")
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;margin-top:-0.6rem;margin-bottom:0.8rem;">'
                'Create a new column by grouping a numeric column into labeled ranges '
                '(e.g. Age → Children / Young / Adult). Only numeric columns with '
                '<strong>no missing values</strong> are eligible -- fill missing values above first.</p>',
                unsafe_allow_html=True
            )

            df = st.session_state.df
            eligible_cols = [
                c for c in df.select_dtypes(include=np.number).columns
                if df[c].isnull().sum() == 0
            ]

            if not eligible_cols:
                banner("warn", "No eligible numeric columns -- fill missing values in a numeric column above first, or this dataset has no numeric columns.")
            else:
                ac1, ac2 = st.columns(2)
                with ac1:
                    source_col = st.selectbox(
                        "Source Column (numeric, no missing values)",
                        eligible_cols,
                        key="acr_source_col"
                    )
                with ac2:
                    new_col_name = st.text_input(
                        "New Column Name",
                        placeholder="e.g. age_group",
                        key="acr_new_col_name"
                    )

                col_min = float(df[source_col].min())
                col_max = float(df[source_col].max())
                st.caption(f"'{source_col}' ranges from {col_min:g} to {col_max:g} in this dataset — ranges below can't go outside this.")

                if "acr_rows" not in st.session_state or st.session_state.get("acr_rows_for_col") != source_col:
                    st.session_state["acr_rows"] = [{"id": 0, "min": col_min, "max": col_max, "label": ""}]
                    st.session_state["acr_next_id"] = 1
                    st.session_state["acr_rows_for_col"] = source_col

                row_to_delete = None
                for row in st.session_state["acr_rows"]:
                    row["min"] = min(max(float(row["min"]), col_min), col_max)
                    row["max"] = min(max(float(row["max"]), col_min), col_max)

                    rc1, rc2, rc3, rc4 = st.columns([2, 2, 3, 1])
                    with rc1:
                        row["min"] = st.number_input(
                            "Min", value=row["min"],
                            min_value=col_min, max_value=col_max,
                            key=f"acr_min_{row['id']}"
                        )
                    with rc2:
                        row["max"] = st.number_input(
                            "Max", value=row["max"],
                            min_value=col_min, max_value=col_max,
                            key=f"acr_max_{row['id']}"
                        )
                    with rc3:
                        row["label"] = st.text_input(
                            "Label", value=row["label"], placeholder="e.g. Children", key=f"acr_label_{row['id']}"
                        )
                    with rc4:
                        st.markdown('<div style="height:1.7rem;"></div>', unsafe_allow_html=True)
                        if st.button(
                            "🗑", key=f"acr_del_{row['id']}", use_container_width=True,
                            disabled=len(st.session_state["acr_rows"]) <= 1
                        ):
                            row_to_delete = row["id"]

                if row_to_delete is not None:
                    st.session_state["acr_rows"] = [
                        r for r in st.session_state["acr_rows"] if r["id"] != row_to_delete
                    ]
                    st.rerun()

                if st.button("+ Add Range", key="acr_add_row"):
                    new_id = st.session_state["acr_next_id"]
                    st.session_state["acr_rows"].append({"id": new_id, "min": col_min, "max": col_max, "label": ""})
                    st.session_state["acr_next_id"] += 1
                    st.rerun()

                divider()

                # ── Validation (runs every render, before the Apply button) ──
                errors = []
                name_clean = new_col_name.strip()
                if not name_clean:
                    errors.append("Enter a name for the new column.")
                elif name_clean in df.columns:
                    errors.append(f"Column '{name_clean}' already exists -- choose a different name.")

                ranges = st.session_state["acr_rows"]
                for i, r in enumerate(ranges, start=1):
                    if r["min"] > r["max"]:
                        errors.append(f"Range {i}: Min ({r['min']:g}) cannot be greater than Max ({r['max']:g}).")
                    if not r["label"].strip():
                        errors.append(f"Range {i}: Label cannot be empty.")

                    if r["min"] < col_min or r["max"] > col_max:
                        errors.append(
                            f"Range {i}: must stay within '{source_col}''s actual range "
                            f"({col_min:g}–{col_max:g})."
                        )

                sorted_ranges = sorted(ranges, key=lambda r: r["min"])
                for i in range(len(sorted_ranges) - 1):
                    if sorted_ranges[i]["max"] >= sorted_ranges[i + 1]["min"]:
                        errors.append(
                            f"Ranges overlap: '{sorted_ranges[i]['label'] or '(unlabeled)'}' "
                            f"({sorted_ranges[i]['min']:g}-{sorted_ranges[i]['max']:g}) overlaps with "
                            f"'{sorted_ranges[i + 1]['label'] or '(unlabeled)'}' "
                            f"({sorted_ranges[i + 1]['min']:g}-{sorted_ranges[i + 1]['max']:g})."
                        )
                        break

                if errors:
                    for e in errors:
                        banner("err", e)

                if st.button(
                    "Add Column", key="acr_apply_btn", type="primary",
                    disabled=len(errors) > 0, use_container_width=True
                ):
                    conditions = [(df[source_col] >= r["min"]) & (df[source_col] <= r["max"]) for r in ranges]
                    choices    = [r["label"].strip() for r in ranges]

                    new_values = np.select(conditions, choices, default="")
                    new_series = pd.Series(new_values, index=df.index).replace("", np.nan)

                    push_cleaning_history()
                    st.session_state.df[name_clean] = new_series
                    uncovered = int(new_series.isna().sum())

                    msg = f"Column '{name_clean}' created from '{source_col}'"
                    if uncovered > 0:
                        msg += f" -- {uncovered} row(s) fell outside all ranges and were left blank"

                    st.session_state["last_action"] = msg
                    st.session_state.pop("acr_rows", None)
                    st.session_state.pop("acr_rows_for_col", None)
                    st.rerun()

            divider()

            # ── Delete Column ──
            eyebrow("Delete a Column")

            df = st.session_state.df

            if len(df.columns) == 0:
                banner("warn", "No columns remaining to delete. Reset the dataset first.")
            elif len(df.columns) == 1:
                banner("warn", f"Only one column (<strong>{df.columns[0]}</strong>) remains -- deleting it would leave an empty dataset, so this is disabled.")
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    del_col = st.selectbox("Select column to delete", df.columns.tolist(), key="fill_del_col")
                with col2:
                    st.markdown('<div style="height:1.7rem;"></div>', unsafe_allow_html=True)
                    if st.button("Delete Column", key="fill_del_btn", use_container_width=True):
                        push_cleaning_history()
                        st.session_state.df = st.session_state.df.drop(columns=[del_col])
                        st.session_state["last_action"] = f"'{del_col}' column deleted"
                        st.rerun()

            divider()

            # ── Undo / Reset ──
            eyebrow("Undo / Reset")
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;margin-top:-0.6rem;margin-bottom:0.8rem;">'
                'Undo reverses your last fill / add-column / delete action on this tab. Reset always restores the original uploaded file.</p>',
                unsafe_allow_html=True
            )

            cleaning_history = st.session_state.get("cleaning_history", [])
            undo_col, reset_col = st.columns(2)

            with undo_col:
                if st.button(
                    f"↩ Undo Last Action ({len(cleaning_history)} step{'s' if len(cleaning_history) != 1 else ''})",
                    key="undo_btn",
                    disabled=len(cleaning_history) == 0,
                    use_container_width=True
                ):
                    st.session_state.df = st.session_state["cleaning_history"].pop()
                    st.session_state["last_action"] = "Last action undone"
                    st.rerun()

            with reset_col:
                if st.button("🔄 Reset to Original Dataset", key="fill_reset_btn", use_container_width=True):
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state["cleaning_history"] = []
                    st.session_state.pop("acr_rows", None)
                    st.session_state.pop("acr_rows_for_col", None)
                    st.session_state["last_action"] = "Dataset reset to original"
                    st.rerun()

            divider()

            # ── Download ──
            eyebrow("Download Cleaned Dataset")
            csv = st.session_state.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                key="cleaning_download"
            )

        # ---------- DUPLICATE ROWS ----------
        with eda_tab5:
            df = st.session_state.df

            def push_duplicate_history():
                history = st.session_state.setdefault("duplicate_history", [])
                history.append(st.session_state.df.copy())
                if len(history) > 20:
                    history.pop(0)

            # ── Check Mode ──
            eyebrow("Check Duplicates")

            check_mode = st.radio(
                "Check duplicates based on:",
                ["All Columns", "Specific Column"],
                key="dup_mode",
                horizontal=True
            )

            subset = None
            if check_mode == "Specific Column":
                selected_col = st.selectbox(
                    "Select Column (e.g. ID, Order No, etc.)",
                    df.columns.tolist(),
                    key="dup_col"
                )
                subset = [selected_col]

            divider()

            # ── Stats ──
            duplicate_count = int(df.duplicated(subset=subset).sum())
            total_rows      = len(df)
            dup_percent     = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0

            dup_cls = "yellow" if duplicate_count > 0 else "green"

            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card" style="--accent:#388bfd">
                    <div class="stat-label">Total Rows</div>
                    <div class="stat-value blue">{total_rows:,}</div>
                </div>
                <div class="stat-card" style="--accent:{'#d29922' if duplicate_count > 0 else '#3fb950'}">
                    <div class="stat-label">Duplicate Rows</div>
                    <div class="stat-value {dup_cls}">{duplicate_count:,}</div>
                </div>
                <div class="stat-card" style="--accent:{'#d29922' if duplicate_count > 0 else '#3fb950'}">
                    <div class="stat-label">Duplicate %</div>
                    <div class="stat-value {dup_cls}">{dup_percent}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Preview ──
            if duplicate_count == 0:
                banner("ok", "No duplicate rows found.")
            else:
                banner("warn", f"{duplicate_count:,} duplicate rows found.")
                with st.expander("Preview Duplicate Rows"):
                    st.dataframe(
                        df[df.duplicated(subset=subset)].head(50),
                        use_container_width=True
                    )
                    if duplicate_count > 50:
                        st.caption(f"Showing 50 of {duplicate_count:,} duplicate rows.")

            # ── Message ──
            if "dup_msg" in st.session_state:
                banner("ok", st.session_state["dup_msg"])
                del st.session_state["dup_msg"]

            divider()

            # ── Actions ──
            eyebrow("Actions")

            btn1, btn2 = st.columns(2)

            with btn1:
                if st.button(
                    "🗑 Delete All Duplicates",
                    key="delete_dup",
                    disabled=duplicate_count == 0,
                    use_container_width=True
                ):
                    push_duplicate_history()
                    before = len(st.session_state.df)
                    st.session_state.df = st.session_state.df.drop_duplicates(subset=subset).reset_index(drop=True)
                    after = len(st.session_state.df)
                    st.session_state["dup_msg"] = f"{before - after:,} duplicate rows removed"
                    st.rerun()

            with btn2:
                if st.button("🔄 Reset to Original Dataset", key="reset_dup", use_container_width=True):
                    st.session_state.df = st.session_state.original_df.copy()
                    st.session_state["duplicate_history"] = []
                    st.session_state["dup_msg"] = "Dataset reset to original"
                    st.rerun()

            # ── Undo ──
            duplicate_history = st.session_state.get("duplicate_history", [])
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;margin-top:0.6rem;margin-bottom:0.5rem;">'
                'Undo reverses your last delete-duplicates action on this tab. Reset always restores the original uploaded file.</p>',
                unsafe_allow_html=True
            )
            if st.button(
                f"↩ Undo Last Action ({len(duplicate_history)} step{'s' if len(duplicate_history) != 1 else ''})",
                key="undo_dup_btn",
                disabled=len(duplicate_history) == 0,
                use_container_width=True
            ):
                st.session_state.df = st.session_state["duplicate_history"].pop()
                st.session_state["dup_msg"] = "Last action undone"
                st.rerun()

            divider()

            # ── Download ──
            eyebrow("Download Dataset")
            csv = st.session_state.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                key="download_dup"
            )

        # ---------- VISUALIZATION ----------
        with eda_tab6:
            df = st.session_state.df
            from matplotlib.ticker import FuncFormatter

            # ── column pools ──
            numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols      = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            x_axis_cols   = cat_cols + datetime_cols 

            AGG_FUNCS = {
                "Sum": "sum", "Mean": "mean", "Median": "median",
                "Min": "min", "Max": "max", "Count": "count",
            }
            PALETTE = ["#388bfd", "#3fb950", "#f78166", "#d29922", "#bc8cff",
                       "#79c0ff", "#56d364", "#e3b341", "#ff7b72", "#8b949e"]
            MAX_CATEGORIES        = 50 
            MAX_LEGEND_CATEGORIES = 15   

            # ── shared helpers (scoped to this tab -- do not shadow global ones) ──
            def fmt_large(value, _pos=None):
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return str(value)
                a = abs(v)
                if a >= 1_000_000_000: return f"{v/1_000_000_000:.1f}B"
                if a >= 1_000_000:     return f"{v/1_000_000:.1f}M"
                if a >= 1_000:         return f"{v/1_000:.1f}K"
                return f"{v:g}"

            def style_chart(fig, ax, rotate_x=False, horizontal=False):
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors="#8b949e", labelsize=7.5)
                ax.set_xlabel(ax.get_xlabel(), color="#8b949e", fontsize=8.5)
                ax.set_ylabel(ax.get_ylabel(), color="#8b949e", fontsize=8.5)
                if ax.get_title():
                    ax.set_title(ax.get_title(), color="#f0f6fc", fontsize=10.5, fontweight="bold", pad=10)
                for s in ax.spines.values():
                    s.set_visible(False)
                if horizontal:
                    ax.xaxis.grid(True, color="#161b22", linewidth=0.6)
                else:
                    ax.yaxis.grid(True, color="#161b22", linewidth=0.6)
                if rotate_x:
                    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

            def clean_for(cols):
                sub = df[cols].dropna()
                return sub, len(df) - len(sub)

            def warn_if_dropped(dropped, total):
                if total > 0 and dropped / total > 0.3:
                    banner("warn", f"{dropped:,} of {total:,} rows were excluded due to missing values in the selected columns.")

            def top_n_and_sort(series, sort_option, top_n_option):
                s = series.copy()
                if sort_option == "Descending":
                    s = s.sort_values(ascending=False)
                elif sort_option == "Ascending":
                    s = s.sort_values(ascending=True)
                s = s.head(int(top_n_option))
                return s

            def low_cardinality_cat_cols(exclude_col=None, max_unique=MAX_LEGEND_CATEGORIES):
                """Categorical columns safe to use for a legend/hue/color-by dropdown --
                excludes columns whose cardinality would blow up the chart (e.g. an ID column
                with thousands of unique values) instead of only truncating after the fact."""
                candidates = [c for c in cat_cols if c != exclude_col]
                safe    = [c for c in candidates if 2 <= df[c].nunique() <= max_unique]
                skipped = [c for c in candidates if c not in safe]
                return safe, skipped

            # ── Chart type selector ──
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot",
                 "Histogram", "Box Plot", "Pie Chart"],
                key="viz_chart_type"
            )

            divider()
            eyebrow("Configure")

            config_ok = True
            cfg = {}

            # ============ HISTOGRAM ============
            if chart_type == "Histogram":
                if not numeric_cols:
                    banner("warn", "No numeric columns available for a histogram.")
                    config_ok = False
                else:
                    cfg["col"]  = st.selectbox("Numerical Column", numeric_cols, key="viz_hist_col")
                    cfg["bins"] = st.slider("Number of Bins", 5, 100, 30, key="viz_hist_bins")

            # ============ BOX PLOT ============
            elif chart_type == "Box Plot":
                if not numeric_cols:
                    banner("warn", "No numeric columns available for a box plot.")
                    config_ok = False
                else:
                    b1, b2 = st.columns(2)
                    with b1:
                        cfg["value_col"] = st.selectbox("Numerical Column", numeric_cols, key="viz_box_val")
                    with b2:
                        group_sel = st.selectbox("Group By (optional)", ["None"] + cat_cols, key="viz_box_group")
                        cfg["group_col"] = None if group_sel == "None" else group_sel

            # ============ SCATTER PLOT ============
            elif chart_type == "Scatter Plot":
                if len(numeric_cols) < 2:
                    banner("warn", "Need at least 2 numeric columns for a scatter plot.")
                    config_ok = False
                else:
                    s1, s2 = st.columns(2)
                    with s1:
                        cfg["x_col"] = st.selectbox("X-axis (numeric)", numeric_cols, key="viz_sc_x")
                    with s2:
                        y_choices = [c for c in numeric_cols if c != cfg["x_col"]] or numeric_cols
                        cfg["y_col"] = st.selectbox("Y-axis (numeric)", y_choices, key="viz_sc_y")

                    color_options, color_skipped = low_cardinality_cat_cols()

                    s3, s4 = st.columns(2)
                    with s3:
                        color_sel = st.selectbox("Color by (optional)", ["None"] + color_options, key="viz_sc_color")
                        cfg["color_col"] = None if color_sel == "None" else color_sel
                    with s4:
                        size_opts = ["None"] + [c for c in numeric_cols if c not in (cfg["x_col"], cfg["y_col"])]
                        size_sel  = st.selectbox("Size by (optional)", size_opts, key="viz_sc_size")
                        cfg["size_col"] = None if size_sel == "None" else size_sel

                    if color_skipped:
                        st.caption(f"Not shown for Color by (too many categories, >{MAX_LEGEND_CATEGORIES}): {', '.join(color_skipped)}")

            # ============ PIE CHART ============
            elif chart_type == "Pie Chart":
                if not cat_cols:
                    banner("warn", "No categorical columns available for a pie chart.")
                    config_ok = False
                else:
                    p1, p2 = st.columns(2)
                    with p1:
                        cfg["cat_col"] = st.selectbox("Category", cat_cols, key="viz_pie_cat")
                    with p2:
                        value_sel = st.selectbox("Value", ["Count (rows)"] + numeric_cols, key="viz_pie_val")
                        cfg["value_col"] = None if value_sel == "Count (rows)" else value_sel
                    cfg["agg"] = (
                        st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Min", "Max"], key="viz_pie_agg")
                        if cfg["value_col"] else "Count"
                    )

            # ============ BAR / LINE / AREA (shared field layout) ============
            else:
                if not x_axis_cols:
                    banner("warn", "No categorical or datetime columns available for the X-axis. Bar / Line / Area charts group a numeric Y-axis value by a categorical or date X-axis.")
                    config_ok = False
                elif not numeric_cols:
                    banner("warn", "No numeric columns available for the Y-axis.")
                    config_ok = False
                else:
                    v1, v2 = st.columns(2)
                    with v1:
                        cfg["x_col"] = st.selectbox("X-axis (category / date)", x_axis_cols, key=f"viz_xy_x_{chart_type}")
                    with v2:
                        cfg["y_col"] = st.selectbox("Y-axis (numeric)", numeric_cols, key=f"viz_xy_y_{chart_type}")

                    color_options, color_skipped = low_cardinality_cat_cols(exclude_col=cfg["x_col"])

                    v3, v4 = st.columns(2)
                    with v3:
                        cfg["agg"] = st.selectbox("Aggregation", list(AGG_FUNCS.keys()), key=f"viz_xy_agg_{chart_type}")
                    with v4:
                        color_sel = st.selectbox("Color / Legend (optional)", ["None"] + color_options, key=f"viz_xy_color_{chart_type}")
                        cfg["color_col"] = None if color_sel == "None" else color_sel

                    if color_skipped:
                        st.caption(f"Not shown for Color / Legend (too many categories, >{MAX_LEGEND_CATEGORIES}): {', '.join(color_skipped)}")

                    is_dt_x = cfg["x_col"] in datetime_cols

                    if chart_type == "Bar Chart" and not is_dt_x:
                        v5, v6 = st.columns(2)
                        with v5:
                            cfg["sort"] = st.selectbox("Sort", ["Descending", "Ascending", "Original order"], key="viz_bar_sort")
                        with v6:
                            cfg["top_n"] = st.selectbox("Show Top N", ["5", "10", "15", "20"], index=1, key="viz_bar_topn")
                    else:
                        cfg["sort"], cfg["top_n"] = "Original order", str(MAX_CATEGORIES)

            divider()
            generate = st.button("Generate Chart", key="viz_generate_btn", type="primary", disabled=not config_ok)

            if generate and config_ok:
                divider()
                eyebrow("Chart Output")
                with st.spinner("Generating chart..."):
                    try:
                        # ============ HISTOGRAM ============
                        if chart_type == "Histogram":
                            col = cfg["col"]
                            clean = df[[col]].dropna()[col]
                            if clean.empty:
                                banner("warn", f"<strong>{col}</strong> has no non-null values to plot.")
                            else:
                                mean_v, median_v = clean.mean(), clean.median()
                                st.caption(f"n = {len(clean):,}   ·   mean = {mean_v:,.2f}   ·   median = {median_v:,.2f}")
                                fig, ax = plt.subplots(figsize=(7.5, 4))
                                sns.histplot(clean, bins=cfg["bins"], color="#388bfd", alpha=0.85, ax=ax, edgecolor="none")
                                ax.axvline(mean_v, color="#d29922", linewidth=1.2, linestyle="--", label="Mean")
                                ax.axvline(median_v, color="#3fb950", linewidth=1.2, linestyle="--", label="Median")
                                ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117", edgecolor="#21262d")
                                ax.set_xlabel(col); ax.set_ylabel("Frequency")
                                ax.set_title(f"Distribution of {col}")
                                ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                style_chart(fig, ax)
                                plt.tight_layout(); st.pyplot(fig)
                                download_chart(fig, "viz_histogram_download")
                                plt.close(fig)

                        # ============ BOX PLOT ============
                        elif chart_type == "Box Plot":
                            value_col, group_col = cfg["value_col"], cfg["group_col"]
                            if group_col:
                                sub, dropped = clean_for([value_col, group_col])
                                warn_if_dropped(dropped, len(df))
                                if sub.empty:
                                    banner("warn", "No valid data remains after removing missing values.")
                                else:
                                    top_cats = sub[group_col].value_counts().head(15).index.tolist()
                                    sub = sub[sub[group_col].isin(top_cats)]
                                    groups = [sub.loc[sub[group_col] == c, value_col].values for c in top_cats]
                                    fig, ax = plt.subplots(figsize=(max(7, len(top_cats) * 0.9), 4.5))
                                    ax.boxplot(groups, vert=True, patch_artist=True, widths=0.55,
                                               boxprops=dict(facecolor="#161b22", color="#388bfd"),
                                               medianprops=dict(color="#f78166", linewidth=2),
                                               whiskerprops=dict(color="#388bfd"), capprops=dict(color="#388bfd"),
                                               flierprops=dict(marker="o", color="#d29922", markersize=3, alpha=0.5))
                                    ax.set_xticks(range(1, len(top_cats) + 1))
                                    ax.set_xticklabels([str(c) for c in top_cats])
                                    ax.set_ylabel(value_col); ax.set_xlabel(group_col)
                                    ax.set_title(f"{value_col} Distribution by {group_col}")
                                    style_chart(fig, ax, rotate_x=len(top_cats) > 6)
                                    plt.tight_layout(); st.pyplot(fig)
                                    download_chart(fig, "viz_box_download")
                                    plt.close(fig)
                                    if sub[group_col].nunique() < df[group_col].nunique():
                                        st.caption(f"Showing the 15 most frequent categories of '{group_col}'.")
                            else:
                                clean = df[[value_col]].dropna()[value_col]
                                if clean.empty:
                                    banner("warn", f"<strong>{value_col}</strong> has no non-null values to plot.")
                                else:
                                    fig, ax = plt.subplots(figsize=(7, 3.2))
                                    ax.boxplot(clean, vert=False, patch_artist=True, widths=0.5,
                                               boxprops=dict(facecolor="#161b22", color="#388bfd"),
                                               medianprops=dict(color="#f78166", linewidth=2),
                                               whiskerprops=dict(color="#388bfd"), capprops=dict(color="#388bfd"),
                                               flierprops=dict(marker="o", color="#d29922", markersize=3, alpha=0.5))
                                    ax.set_xlabel(value_col); ax.set_yticks([])
                                    ax.set_title(f"{value_col} Distribution")
                                    style_chart(fig, ax, horizontal=True)
                                    plt.tight_layout(); st.pyplot(fig)
                                    download_chart(fig, "viz_box_download")
                                    plt.close(fig)

                        # ============ SCATTER PLOT ============
                        elif chart_type == "Scatter Plot":
                            x_col, y_col = cfg["x_col"], cfg["y_col"]
                            color_col, size_col = cfg["color_col"], cfg["size_col"]
                            needed = [x_col, y_col] + ([color_col] if color_col else []) + ([size_col] if size_col else [])
                            sub, dropped = clean_for(needed)
                            warn_if_dropped(dropped, len(df))
                            if sub.empty:
                                banner("warn", "No valid data remains after removing missing values in the selected columns.")
                            else:
                                sampled = False
                                if len(sub) > 10000:
                                    sub = sub.sample(10000, random_state=42); sampled = True
                                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                                sizes = 18
                                if size_col:
                                    raw = sub[size_col].astype(float)
                                    rng = raw.max() - raw.min()
                                    sizes = 15 + ((raw - raw.min()) / rng * 90 if rng > 0 else 30)
                                if color_col:
                                    for i, cat in enumerate(sub[color_col].unique()):
                                        mask = sub[color_col] == cat
                                        pt_sizes = sizes[mask] if hasattr(sizes, "__len__") else sizes
                                        ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col],
                                                   color=PALETTE[i % len(PALETTE)], label=str(cat),
                                                   alpha=0.65, s=pt_sizes, edgecolors="none")
                                    ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117",
                                              edgecolor="#21262d", title=color_col, title_fontsize=7)
                                else:
                                    ax.scatter(sub[x_col], sub[y_col], color="#388bfd", alpha=0.55, s=sizes, edgecolors="none")
                                ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                                ax.set_title(f"{y_col} vs {x_col}")
                                ax.xaxis.set_major_formatter(FuncFormatter(fmt_large))
                                ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                style_chart(fig, ax)
                                plt.tight_layout(); st.pyplot(fig)
                                if sampled:
                                    st.caption(f"Showing a random sample of 10,000 of {len(df):,} rows for rendering speed.")
                                download_chart(fig, "viz_scatter_download")
                                plt.close(fig)

                        # ============ PIE CHART ============
                        elif chart_type == "Pie Chart":
                            cat_col, value_col, agg_label = cfg["cat_col"], cfg["value_col"], cfg["agg"]
                            series = None
                            if value_col:
                                sub, dropped = clean_for([cat_col, value_col])
                                warn_if_dropped(dropped, len(df))
                                if not sub.empty:
                                    series = sub.groupby(cat_col)[value_col].agg(AGG_FUNCS[agg_label])
                            else:
                                sub, dropped = clean_for([cat_col])
                                warn_if_dropped(dropped, len(df))
                                if not sub.empty:
                                    series = sub[cat_col].value_counts()

                            if series is None or series.empty:
                                banner("warn", f"<strong>{cat_col}</strong> has no values to plot.")
                            else:
                                series = series.sort_values(ascending=False)
                                if len(series) > 8:
                                    series = pd.concat([series.head(8), pd.Series({"Other": series.iloc[8:].sum()})])

                                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                                fig.patch.set_facecolor("#0d1117")
                                wedges, texts, autotexts = ax.pie(
                                    series.values, labels=series.index.astype(str),
                                    colors=PALETTE[:len(series)], autopct="%1.1f%%", startangle=90,
                                    pctdistance=0.72, textprops={"color": "#8b949e", "fontsize": 7.5},
                                    wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
                                )
                                for at in autotexts:
                                    at.set_color("#f0f6fc"); at.set_fontsize(7.5)
                                ax.set_facecolor("#0d1117")
                                metric = value_col if value_col else "Count"
                                ax.set_title(f"{agg_label} of {metric} by {cat_col}",
                                             color="#f0f6fc", fontsize=10.5, fontweight="bold", pad=12)
                                plt.tight_layout(); st.pyplot(fig)
                                download_chart(fig, "viz_pie_download")
                                plt.close(fig)

                        # ============ BAR / LINE / AREA ============
                        else:
                            x_col, y_col, agg_label = cfg["x_col"], cfg["y_col"], cfg["agg"]
                            color_col = cfg["color_col"]
                            needed = [x_col, y_col] + ([color_col] if color_col else [])
                            sub, dropped = clean_for(needed)
                            warn_if_dropped(dropped, len(df))

                            if sub.empty:
                                banner("warn", "No valid data remains after removing missing values in the selected columns.")
                            else:
                                is_dt_x   = x_col in datetime_cols
                                agg_func  = AGG_FUNCS[agg_label]
                                grouped   = None
                                series    = None

                                if color_col:
                                    grouped = sub.groupby([x_col, color_col])[y_col].agg(agg_func).unstack(fill_value=0)
                                    if is_dt_x:
                                        grouped = grouped.sort_index()
                                    else:
                                        order = grouped.sum(axis=1)
                                        grouped = grouped.loc[top_n_and_sort(order, cfg["sort"], cfg["top_n"]).index]
                                        if len(grouped) > MAX_CATEGORIES:
                                            banner("info", f"{len(grouped):,} categories found — showing the top {MAX_CATEGORIES} by total value for readability.")
                                            order = grouped.sum(axis=1)
                                            grouped = grouped.loc[order.sort_values(ascending=False).head(MAX_CATEGORIES).index]
                                else:
                                    series = sub.groupby(x_col)[y_col].agg(agg_func)
                                    if is_dt_x:
                                        series = series.sort_index()
                                    else:
                                        series = top_n_and_sort(series, cfg["sort"], cfg["top_n"])
                                        if len(series) > MAX_CATEGORIES:
                                            banner("info", f"{len(series):,} categories found — showing the top {MAX_CATEGORIES} by value for readability.")
                                            series = series.sort_values(ascending=False).head(MAX_CATEGORIES)

                                n_cats = len(grouped) if grouped is not None else len(series)
                                title  = f"{agg_label} of {y_col} by {x_col}"

                                # ---- BAR ----
                                if chart_type == "Bar Chart":
                                    if color_col:
                                        fig, ax = plt.subplots(figsize=(max(7, n_cats * 0.5), 4.5))
                                        bottom = np.zeros(len(grouped))
                                        labels = [str(v) for v in grouped.index]
                                        for i, cat in enumerate(grouped.columns):
                                            ax.bar(labels, grouped[cat].values, bottom=bottom,
                                                   color=PALETTE[i % len(PALETTE)], label=str(cat), edgecolor="none")
                                            bottom += grouped[cat].values
                                        ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117",
                                                  edgecolor="#21262d", title=color_col, title_fontsize=7)
                                        ax.set_xlabel(x_col); ax.set_ylabel(f"{agg_label} of {y_col}")
                                        ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                        ax.set_title(title)
                                        style_chart(fig, ax, rotate_x=n_cats > 6)
                                    elif is_dt_x:
                                        fig, ax = plt.subplots(figsize=(7.5, 4.5))
                                        ax.bar(series.index, series.values, color="#388bfd", width=0.8, edgecolor="none")
                                        ax.set_xlabel(x_col); ax.set_ylabel(f"{agg_label} of {y_col}")
                                        ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                        ax.set_title(title)
                                        style_chart(fig, ax, rotate_x=True)
                                    else:
                                        max_idx = int(np.argmax(series.values)) if len(series) else 0
                                        colors  = ["#388bfd" if i == max_idx else "#21262d" for i in range(len(series))]
                                        fig, ax = plt.subplots(figsize=(7.5, max(3, len(series) * 0.4)))
                                        ax.barh([str(v) for v in series.index][::-1], series.values[::-1],
                                                color=colors[::-1], edgecolor="none")
                                        ax.set_ylabel(x_col); ax.set_xlabel(f"{agg_label} of {y_col}")
                                        ax.xaxis.set_major_formatter(FuncFormatter(fmt_large))
                                        ax.set_title(title)
                                        style_chart(fig, ax, horizontal=True)
                                    plt.tight_layout(); st.pyplot(fig)
                                    download_chart(fig, "viz_bar_download")
                                    plt.close(fig)

                                # ---- LINE ----
                                elif chart_type == "Line Chart":
                                    fig, ax = plt.subplots(figsize=(7.5, 4.5))
                                    if color_col:
                                        for i, cat in enumerate(grouped.columns):
                                            ax.plot(grouped.index, grouped[cat].values, color=PALETTE[i % len(PALETTE)],
                                                    linewidth=1.6, marker="o", markersize=3, label=str(cat))
                                        ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117",
                                                  edgecolor="#21262d", title=color_col, title_fontsize=7)
                                    else:
                                        ax.plot(series.index, series.values, color="#388bfd", linewidth=1.8, marker="o", markersize=3)
                                    ax.set_xlabel(x_col); ax.set_ylabel(f"{agg_label} of {y_col}")
                                    ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                    ax.set_title(f"{y_col} Trend by {x_col}" if is_dt_x else title)
                                    style_chart(fig, ax, rotate_x=is_dt_x or n_cats > 8)
                                    plt.tight_layout(); st.pyplot(fig)
                                    download_chart(fig, "viz_line_download")
                                    plt.close(fig)

                                # ---- AREA ----
                                else:
                                    fig, ax = plt.subplots(figsize=(7.5, 4.5))
                                    if color_col:
                                        ax.stackplot(grouped.index, [grouped[c].values for c in grouped.columns],
                                                     labels=[str(c) for c in grouped.columns],
                                                     colors=PALETTE[:len(grouped.columns)], alpha=0.85)
                                        ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117",
                                                  edgecolor="#21262d", title=color_col, title_fontsize=7, loc="upper left")
                                    else:
                                        ax.fill_between(series.index, series.values, color="#388bfd", alpha=0.35)
                                        ax.plot(series.index, series.values, color="#388bfd", linewidth=1.6)
                                    ax.set_xlabel(x_col); ax.set_ylabel(f"{agg_label} of {y_col}")
                                    ax.yaxis.set_major_formatter(FuncFormatter(fmt_large))
                                    ax.set_title(f"{y_col} Trend by {x_col}" if is_dt_x else title)
                                    style_chart(fig, ax, rotate_x=is_dt_x or n_cats > 8)
                                    plt.tight_layout(); st.pyplot(fig)
                                    download_chart(fig, "viz_area_download")
                                    plt.close(fig)

                    except Exception as e:
                        banner("err", f"Error generating chart: {str(e)}")


    # ---------------- MODEL LAB ----------------
    with main_tab_model:
        model_tab1, model_tab2, model_tab3 = st.tabs([
            "Encoding", "Feature Importance", "Training"
        ])

        # ================= ENCODING =================
        with model_tab1:
            df       = st.session_state.df
            cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

            if "enc_history" not in st.session_state:
                st.session_state["enc_history"] = []

            if not cat_cols:
                banner("ok", "No categorical columns found — all columns are already encoded.")
            else:
                # ── 1. Categorical Columns Info ──
                eyebrow("Categorical Columns")

                cat_info = pd.DataFrame({
                    "Column"        : cat_cols,
                    "Unique Values" : [df[col].nunique() for col in cat_cols],
                    "Missing Values": [int(df[col].isnull().sum()) for col in cat_cols],
                }).sort_values("Unique Values", ascending=False).reset_index(drop=True)

                st.dataframe(cat_info, use_container_width=True, hide_index=True)
                divider()

                # ── 2. Apply Encoding ──
                eyebrow("Apply Encoding")

                col1, col2 = st.columns(2)
                with col1:
                    selected_col = st.selectbox("Select Column", cat_cols, key="enc_col")
                with col2:
                    method = st.selectbox(
                        "Encoding Method",
                        ["Label Encoding", "One Hot Encoding", "Manual (Ordinal)"],
                        key="enc_method",
                    )

                unique_count = df[selected_col].nunique()
                has_missing  = bool(df[selected_col].isnull().sum() > 0)
                ohe_blocked  = method == "One Hot Encoding" and unique_count > 25
                # NEW: manual ordinal mapping only makes sense for a small,
                # human-typeable set of categories -- same idea as the OHE cap.
                manual_blocked = method == "Manual (Ordinal)" and unique_count > 20

                # info card
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                    f'padding:0.65rem 1rem;font-family:\'Space Mono\',monospace;font-size:0.72rem;'
                    f'color:#8b949e;margin:0.4rem 0 0.7rem 0;">'
                    f'<span style="color:#f0f6fc;">{unique_count}</span> unique values in '
                    f'<span style="color:#388bfd;">{selected_col}</span></div>',
                    unsafe_allow_html=True,
                )

                if has_missing:
                    banner(
                        "err",
                        f"<strong>{selected_col}</strong> has missing values — "
                        f"fill them in the <strong>Data Cleaning</strong> tab first.",
                    )
                if ohe_blocked:
                    banner(
                        "err",
                        f"<strong>{selected_col}</strong> has {unique_count} unique values "
                        f"(max 25 for OHE). Use Label Encoding instead.",
                    )
                if manual_blocked:
                    banner(
                        "err",
                        f"<strong>{selected_col}</strong> has {unique_count} unique values "
                        f"(max 20 for Manual Ordinal Mapping). Use Label Encoding instead.",
                    )

                # ── Manual ordinal input ──
                order = ""
                if method == "Manual (Ordinal)" and not manual_blocked:
                    unique_vals = df[selected_col].dropna().unique().tolist()
                    st.markdown(
                        f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                        f'padding:0.6rem 1rem;font-family:\'Space Mono\',monospace;font-size:0.68rem;'
                        f'color:#8b949e;margin-bottom:0.5rem;">'
                        f'Available values: <span style="color:#d29922;">'
                        f'{", ".join(str(v) for v in unique_vals)}</span></div>',
                        unsafe_allow_html=True,
                    )
                    order = st.text_input(
                        "Enter order — lowest → highest, comma separated",
                        placeholder="e.g. poor, fair, good, excellent",
                        key="ordinal_input",
                    )

                # ── Apply button ──
                is_disabled = has_missing or ohe_blocked or manual_blocked
                if st.button("Apply Encoding", key="enc_btn", disabled=is_disabled):
                    working_df = st.session_state.df.copy()
                    series = working_df[selected_col]
                    encoded_ok = False

                    if method == "Label Encoding":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        working_df[selected_col] = le.fit_transform(series.astype(str))

                        msg = f"Label Encoding applied on <strong>{selected_col}</strong>"
                        encoded_ok = True

                    elif method == "One Hot Encoding":
                        dummies = pd.get_dummies(
                            series,
                            prefix=selected_col,
                            dtype=np.int8
                        )
                        working_df = pd.concat(
                            [working_df.drop(columns=[selected_col]), dummies], axis=1
                        )
                        msg = (f"One Hot Encoding applied on <strong>{selected_col}</strong> "
                               f"→ {dummies.shape[1]} new columns created")
                        encoded_ok = True

                    else:  # Manual (Ordinal)
                        if not order.strip():
                            banner("warn", "Please enter the ordinal order before applying.")
                        else:
                            values = [x.strip() for x in order.split(",")]
                            actual_values = (
                                working_df[selected_col]
                                .dropna()
                                .unique()
                                .tolist()
                            )

                            invalid      = set(values) - set(actual_values)
                            missing_vals = set(actual_values) - set(values)

                            if invalid:
                                banner(
                                    "err",
                                    f"Invalid value(s): <strong>{', '.join(invalid)}</strong> "
                                    f"— check spelling and try again.",
                                )
                            elif missing_vals:
                                banner(
                                    "err",
                                    f"Missing category(s): <strong>{', '.join(missing_vals)}</strong>",
                                )
                            else:
                                mapping = {val: i for i, val in enumerate(values)}
                                working_df[selected_col] = working_df[selected_col].map(mapping)
                                msg = f"Manual Ordinal Encoding applied on <strong>{selected_col}</strong>"
                                encoded_ok = True

                    if encoded_ok:
                        st.session_state["enc_history"].append(st.session_state.df.copy())
                        st.session_state.df         = working_df
                        st.session_state["enc_msg"]  = msg
                        st.rerun()

            # ── success / info banner (after rerun) ──
            if "enc_msg" in st.session_state:
                banner("ok", st.session_state.pop("enc_msg"))

            divider()

            # ── 3. Undo / Reset — scoped ONLY to encoding actions ──
            eyebrow("Undo / Reset")
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;margin-top:-0.6rem;margin-bottom:0.8rem;">'
                'Undo reverses your last encoding action. Reset removes all encodings applied '
                'in this tab and restores the dataset exactly as it was when you first opened '
                'Model Lab — it does not touch your EDA cleaning.</p>',
                unsafe_allow_html=True,
            )

            enc_history = st.session_state.get("enc_history", [])
            undo_c, reset_c = st.columns(2)

            with undo_c:
                if st.button(
                    f"↩ Undo Last Encoding ({len(enc_history)} step{'s' if len(enc_history) != 1 else ''})",
                    key="enc_undo_btn",
                    disabled=len(enc_history) == 0,
                    use_container_width=True,
                ):
                    st.session_state.df = st.session_state["enc_history"].pop()
                    st.session_state["enc_msg"] = "Last encoding undone"
                    st.rerun()

            with reset_c:
                if st.button(
                    "🔄 Reset Encoding",
                    key="enc_reset_btn",
                    disabled=len(enc_history) == 0,
                    use_container_width=True,
                ):
                    st.session_state.df = st.session_state["enc_history"][0].copy()
                    st.session_state["enc_history"] = []
                    st.session_state["enc_msg"]     = "Encoding reset — cleaning/other tabs unaffected"
                    st.rerun()

            divider()

            # ── 4. Download ──
            eyebrow("Download Encoded Dataset")

            rows, cols    = st.session_state.df.shape
            cat_remaining = st.session_state.df.select_dtypes(include="object").shape[1]

            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;'
                f'padding:0.85rem 1.2rem;margin-bottom:0.9rem;font-family:\'Space Mono\',monospace;'
                f'font-size:0.72rem;color:#8b949e;">'
                f'<span style="color:#f0f6fc;">{rows:,}</span> rows &nbsp;·&nbsp; '
                f'<span style="color:#f0f6fc;">{cols}</span> columns &nbsp;·&nbsp; '
                f'<span style="color:{"#d29922" if cat_remaining else "#3fb950"};">'
                f'{cat_remaining} categorical column{"s" if cat_remaining != 1 else ""} remaining'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Encoded CSV",
                data=csv_bytes,
                file_name="encoded_data.csv",
                mime="text/csv",
                key="enc_download",
            )
 
        # ================= FEATURE IMPORTANCE =================
        with model_tab2:
            df          = st.session_state.df
            non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
            num_cols    = df.select_dtypes(include=np.number).columns.tolist()

            # ── Guards ──
            if non_numeric:
                banner(
                    "warn",
                    f"Please encode these columns first in the <strong>Encoding</strong> tab: "
                    f"<code>{'</code>, <code>'.join(non_numeric)}</code>",
                )
            elif len(num_cols) < 2:
                banner(
                    "warn",
                    "At least <strong>2 numeric columns</strong> are required to run "
                    "Feature Importance.",
                )
            else:
                # ── 1. Configuration ──
                eyebrow("Configuration")

                cfg1, cfg2 = st.columns(2)
                with cfg1:
                    target_col = st.selectbox(
                        "Select Target Column (Y)", num_cols, key="fi_target"
                    )
                feature_cols = [c for c in num_cols if c != target_col]

                MAX_ROWS = 20_000
                is_large = len(df) > MAX_ROWS

                with cfg2:
                    n_estimators = st.slider(
                        "Number of Trees", min_value=50, max_value=500,
                        value=100, step=50, key="fi_n_est",
                    )

                target_unique = df[target_col].nunique()
                is_classifier = target_unique <= 10
                model_label   = "Classifier" if is_classifier else "Regressor"
                model_color   = "#d29922" if is_classifier else "#388bfd"

                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
                    f'padding:0.65rem 1rem;font-family:\'Space Mono\',monospace;font-size:0.72rem;'
                    f'color:#8b949e;margin:0.5rem 0 0.4rem 0;">'
                    f'Auto-detected &nbsp;→&nbsp; '
                    f'<span style="color:{model_color};font-weight:600;">'
                    f'Random Forest {model_label}</span>'
                    f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'<span style="color:#f0f6fc;">{len(feature_cols)}</span> features'
                    f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'<span style="color:#f0f6fc;">{target_unique}</span> unique target values'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if is_large:
                    banner(
                        "warn",
                        f"Large dataset ({len(df):,} rows) — will use "
                        f"{MAX_ROWS:,} random samples for speed.",
                    )

                # edge case: only 1 feature col
                if len(feature_cols) == 0:
                    banner(
                        "err",
                        "Only 1 numeric column in dataset — need at least "
                        "<strong>2 columns</strong> (1 feature + 1 target).",
                    )
                else:
                    divider()

                    # ── 2. Run ──
                    eyebrow("Run Feature Importance")

                    if st.button("▶ Run Feature Importance", key="fi_btn"):
                        data = df[feature_cols + [target_col]].dropna()

                        if data.shape[0] < 10:
                            banner(
                                "err",
                                "Not enough clean rows after dropping missing values "
                                "(need ≥ 10).",
                            )
                        else:
                            if len(data) > MAX_ROWS:
                                data = data.sample(MAX_ROWS, random_state=42)

                            X = data[feature_cols]
                            y = data[target_col]

                            with st.spinner("Training Random Forest… this may take a moment."):
                                if is_classifier:
                                    from sklearn.ensemble import RandomForestClassifier
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators, random_state=42, n_jobs=-1
                                    )
                                else:
                                    from sklearn.ensemble import RandomForestRegressor
                                    model = RandomForestRegressor(
                                        n_estimators=n_estimators, random_state=42, n_jobs=-1
                                    )
                                model.fit(X, y)

                            importance_df = pd.DataFrame({
                                "Feature"         : feature_cols,
                                "Importance Score": np.round(model.feature_importances_, 4),
                            }).sort_values("Importance Score", ascending=False).reset_index(drop=True)
                            importance_df["Rank"]         = range(1, len(importance_df) + 1)
                            importance_df["Cumulative %"] = (
                                importance_df["Importance Score"].cumsum() * 100
                            ).round(2)

                            st.session_state["importance_df"]  = importance_df
                            st.session_state["fi_target_used"] = target_col
                            st.session_state["fi_model_label"] = model_label
                            st.session_state["fi_rows_used"]   = len(data)

                    # ── 3. Results ──
                    if "importance_df" in st.session_state:
                        importance_df = st.session_state["importance_df"]
                        divider()

                        eyebrow(
                            f"Results — Target: {st.session_state['fi_target_used']}  "
                            f"({st.session_state['fi_model_label']})"
                        )

                        # top-3 cards
                        top_n     = min(3, len(importance_df))
                        card_cols = st.columns(top_n)
                        medal     = ["🥇", "🥈", "🥉"]
                        for i, st_col in enumerate(card_cols):
                            row = importance_df.iloc[i]
                            st_col.markdown(
                                f'<div style="background:#0d1117;border:1px solid #21262d;'
                                f'border-top:3px solid #388bfd;border-radius:10px;'
                                f'padding:0.9rem 1rem;text-align:center;">'
                                f'<div style="font-size:1.3rem;">{medal[i]}</div>'
                                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
                                f'color:#8b949e;text-transform:uppercase;letter-spacing:0.1em;'
                                f'margin:0.3rem 0 0.2rem 0;">#{int(row["Rank"])}</div>'
                                f'<div style="font-size:0.85rem;font-weight:600;color:#f0f6fc;'
                                f'word-break:break-all;">{row["Feature"]}</div>'
                                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.78rem;'
                                f'color:#388bfd;margin-top:0.3rem;">{row["Importance Score"]}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("<br>", unsafe_allow_html=True)

                        row_height = 35
                        header_h   = 38
                        table_h    = min(600, header_h + len(importance_df) * row_height)
                        st.dataframe(
                            importance_df,
                            use_container_width=True,
                            hide_index=True,
                            height=table_h,
                        )

                        rows_used = st.session_state.get("fi_rows_used", "?")
                        st.caption(f"Trained on {rows_used:,} rows · {n_estimators} trees")

                        divider()

                        # ── 4. Chart ──
                        eyebrow("Importance Chart")

                        fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.42)))
                        fig.patch.set_facecolor("#0d1117")
                        ax.set_facecolor("#0d1117")

                        n          = len(importance_df)
                        alphas     = np.linspace(1.0, 0.35, n)
                        bar_colors = [(56/255, 139/255, 253/255, a) for a in alphas]

                        bars = ax.barh(
                            importance_df["Feature"][::-1],
                            importance_df["Importance Score"][::-1],
                            color=bar_colors[::-1],
                            edgecolor="none",
                        )
                        for bar, val in zip(bars, importance_df["Importance Score"][::-1]):
                            ax.text(
                                bar.get_width() + 0.001,
                                bar.get_y() + bar.get_height() / 2,
                                f"{val:.4f}",
                                va="center", ha="left",
                                color="#8b949e", fontsize=7,
                            )

                        ax.set_xlabel("Importance Score", color="#8b949e", fontsize=8)
                        ax.tick_params(colors="#8b949e", labelsize=7)
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, key="fi_chart_download")
                        plt.close(fig)

        # ================= MODEL TRAINING =================
        with model_tab3:
            df           = st.session_state.df
            has_missing  = df.isnull().sum().sum() > 0
            non_numeric  = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
            too_few_cols = len(df.columns) < 2

            if has_missing:
                banner("err", "Dataset has missing values — handle them in the <strong>Fill Missing Values</strong> tab first.")
            elif non_numeric:
                banner("err", f"Non-numeric columns found: <code>{'</code>, <code>'.join(non_numeric)}</code> — encode them in the <strong>Encoding</strong> tab first.")
            elif too_few_cols:
                banner("err", "Dataset needs at least <strong>2 columns</strong> (1 feature + 1 target) to train a model.")
            else:
                df = df.replace({True: 1, False: 0})

                MAX_ROWS = 20_000
                # FIX: was 15, now 10 — matches Feature Importance tab's
                # is_classifier threshold so the same target column is
                # classified the same way in both tabs.
                CLASSIFICATION_UNIQUE_THRESHOLD = 10

                # ── 1. Target + Task Detection ──
                eyebrow("Target Column")

                target = st.selectbox("Select Target (Y)", df.columns, key="train_target")
                X = df.drop(columns=[target])
                y = df[target]

                if X.shape[1] == 0:
                    banner("err", "No feature columns left after selecting target — dataset needs at least <strong>2 columns</strong>.")
                else:
                    task_type  = "Classification" if (y.dtype == object or y.nunique() <= CLASSIFICATION_UNIQUE_THRESHOLD) else "Regression"
                    task_color = "#d29922" if task_type == "Classification" else "#388bfd"
                    is_large   = len(X) > MAX_ROWS
                    is_highdim = X.shape[1] > 100

                    st.markdown(f"""
                    <div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;
                                padding:0.7rem 1.1rem;font-family:'Space Mono',monospace;font-size:0.72rem;
                                color:#8b949e;margin:0.5rem 0;">
                        Task &nbsp;→&nbsp; <span style="color:{task_color};font-weight:600;">{task_type}</span>
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        <span style="color:#f0f6fc;">{X.shape[1]}</span> features
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        <span style="color:#f0f6fc;">{len(X):,}</span> rows
                        &nbsp;&nbsp;·&nbsp;&nbsp;
                        <span style="color:#f0f6fc;">{y.nunique()}</span> unique target values
                    </div>
                    """, unsafe_allow_html=True)

                    if is_large:
                        banner("warn", f"Large dataset ({len(X):,} rows) — training will use {MAX_ROWS:,} random samples.")
                    if is_highdim:
                        banner("warn", f"High-dimensional data ({X.shape[1]} features) — PCA will auto-apply (90% variance retained).")

                    class_imbalance_warning = None
                    if task_type == "Classification":
                        class_counts = y.value_counts()
                        if (class_counts < 2).any():
                            class_imbalance_warning = "Some classes have only 1 sample — stratified splitting will be skipped for this run."

                    divider()

                    # ── 2. Train / Test Split ──
                    eyebrow("Train / Test Split")
                    test_size = st.slider(
                        "Test Set Size", min_value=0.10, max_value=0.40,
                        value=0.20, step=0.05, key="train_test_size",
                        help="Fraction of data held out for evaluation",
                    )
                    train_rows = int(min(len(X), MAX_ROWS) * (1 - test_size))
                    test_rows  = int(min(len(X), MAX_ROWS) * test_size)
                    st.caption(f"~{train_rows:,} training rows  ·  ~{test_rows:,} test rows")

                    if class_imbalance_warning:
                        banner("warn", class_imbalance_warning)

                    divider()

                    # ── 3. Model Selection ──
                    eyebrow("Select Model")

                    if task_type == "Regression":
                        model_list = ["Linear Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
                    else:
                        model_list = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]

                    if XGBOOST_OK:
                        model_list.append("XGBoost")
                    if LIGHTGBM_OK:
                        model_list.append("LightGBM")

                    if not XGBOOST_OK or not LIGHTGBM_OK:
                        missing_libs = []
                        if not XGBOOST_OK:
                            missing_libs.append("xgboost")
                        if not LIGHTGBM_OK:
                            missing_libs.append("lightgbm")
                        banner("info", f"{' and '.join(missing_libs)} not installed — those models are hidden. Run <code>pip install {' '.join(missing_libs)}</code> to enable them.")

                    model_name = st.selectbox("Model", model_list, key="model_select")

                    # ── 4. Hyperparameter Tuning ──
                    hp = {}
                    hyper_models = ["Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM"]

                    if model_name in hyper_models:
                        enable_tuning = st.toggle("Enable Hyperparameter Tuning", value=False, key="hp_toggle")

                        if enable_tuning:
                            st.markdown("<br>", unsafe_allow_html=True)

                            if model_name == "Decision Tree":
                                c1, c2, c3 = st.columns(3)
                                hp["max_depth"] = c1.slider("Max Depth", 1, 20, 5, help="Higher = more complex, risk of overfitting")
                                hp["min_samples_split"] = c2.slider("Min Samples Split", 2, 20, 5, help="Min samples needed to split a node")
                                hp["min_samples_leaf"] = c3.slider("Min Samples Leaf", 1, 20, 2, help="Min samples at a leaf node")
                                st.caption("Tip: Max Depth 3–8 is usually best. Too high → overfitting.")

                            elif model_name == "Random Forest":
                                c1, c2, c3, c4 = st.columns(4)
                                hp["n_estimators"] = c1.slider("N Estimators", 50, 500, 150, step=50, help="More trees = better but slower")
                                hp["max_depth"] = c2.slider("Max Depth", 1, 20, 7, help="Max depth of each tree")
                                hp["min_samples_split"] = c3.slider("Min Samples Split", 2, 20, 5)
                                hp["min_samples_leaf"] = c4.slider("Min Samples Leaf", 1, 20, 2)
                                st.caption("Tip: Start with 100–200 trees. Max Depth 5–10 works well.")

                            elif model_name == "SVM":
                                c1, c2, c3 = st.columns(3)
                                hp["C"] = c1.select_slider(
                                    "C (Regularization)",
                                    options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                                    value=1.0, help="Higher C = less regularization",
                                )
                                hp["kernel"] = c2.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], help="rbf works best for most cases")
                                hp["gamma"] = c3.selectbox("Gamma", ["scale", "auto"], help="scale = 1/(n_features * X.var())")
                                st.caption("Tip: rbf kernel with C=1.0 is a safe start.")

                            elif model_name == "XGBoost":
                                c1, c2, c3, c4 = st.columns(4)
                                hp["n_estimators"] = c1.slider("N Estimators", 50, 500, 150, step=50)
                                hp["max_depth"] = c2.slider("Max Depth", 1, 15, 6)
                                hp["learning_rate"] = c3.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.1)
                                hp["subsample"] = c4.slider("Subsample", 0.5, 1.0, 1.0, step=0.1, help="Fraction of rows used per tree")
                                st.caption("Tip: Lower learning rate + more trees usually generalizes better.")

                            elif model_name == "LightGBM":
                                c1, c2, c3, c4 = st.columns(4)
                                hp["n_estimators"] = c1.slider("N Estimators", 50, 500, 150, step=50)
                                hp["max_depth"] = c2.slider("Max Depth", -1, 15, -1, help="-1 = no limit")
                                hp["learning_rate"] = c3.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.1)
                                hp["num_leaves"] = c4.slider("Num Leaves", 7, 255, 31)
                                st.caption("Tip: num_leaves should generally be less than 2^max_depth.")

                        else:
                            if model_name == "Decision Tree":
                                hp = {"max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2}
                            elif model_name == "Random Forest":
                                hp = {"n_estimators": 150, "max_depth": 7, "min_samples_split": 5, "min_samples_leaf": 2}
                            elif model_name == "SVM":
                                hp = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
                            elif model_name == "XGBoost":
                                hp = {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "subsample": 1.0}
                            elif model_name == "LightGBM":
                                hp = {"n_estimators": 150, "max_depth": -1, "learning_rate": 0.1, "num_leaves": 31}

                    divider()

                    # ── 5. Train Button ──
                    eyebrow("Train")

                    min_rows_needed = max(10, int(1 / test_size) + 1)
                    if len(X) < min_rows_needed:
                        banner("err", f"Not enough rows ({len(X)}) for a {int(test_size*100)}% test split — need at least <strong>{min_rows_needed}</strong> rows.")
                    else:
                        if st.button("▶ Train Model", key="train_btn", type="primary"):
                            X_s = X.copy()
                            y_s = y.copy()

                            if len(X_s) > MAX_ROWS:
                                X_s = X_s.sample(MAX_ROWS, random_state=42)
                                y_s = y_s.loc[X_s.index]

                            with st.spinner("Training… please wait."):
                                from sklearn.model_selection import train_test_split, cross_val_score
                                from sklearn.preprocessing   import StandardScaler
                                from sklearn.decomposition   import PCA
                                import time

                                start = time.time()

                                stratify_arg = None
                                if task_type == "Classification":
                                    vc = y_s.value_counts()
                                    if (vc >= 2).all():
                                        stratify_arg = y_s

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_s, y_s, test_size=test_size, random_state=42, stratify=stratify_arg
                                )

                                needs_scale = model_name in ["KNN", "SVM", "Linear Regression", "Logistic Regression"]
                                scaler = None
                                if needs_scale or is_highdim:
                                    scaler  = StandardScaler()
                                    X_train = scaler.fit_transform(X_train)
                                    X_test  = scaler.transform(X_test)
                                else:
                                    X_train = X_train.values
                                    X_test  = X_test.values

                                pca         = None
                                pca_n_after = None
                                if is_highdim:
                                    pca         = PCA(n_components=0.90, random_state=42)
                                    X_train     = pca.fit_transform(X_train)
                                    X_test      = pca.transform(X_test)
                                    pca_n_after = X_train.shape[1]

                                if model_name == "Linear Regression":
                                    from sklearn.linear_model import LinearRegression
                                    model = LinearRegression()

                                elif model_name == "Logistic Regression":
                                    from sklearn.linear_model import LogisticRegression
                                    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

                                elif model_name == "KNN":
                                    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                                    k = min(5, len(X_train))
                                    model = (
                                        KNeighborsClassifier(n_neighbors=k, weights="distance")
                                        if task_type == "Classification"
                                        else KNeighborsRegressor(n_neighbors=k, weights="distance")
                                    )

                                elif model_name == "SVM":
                                    from sklearn.svm import SVC, SVR
                                    model = (
                                        SVC(kernel=hp["kernel"], C=hp["C"], gamma=hp["gamma"], probability=True)
                                        if task_type == "Classification"
                                        else SVR(kernel=hp["kernel"], C=hp["C"], gamma=hp["gamma"])
                                    )

                                elif model_name == "Decision Tree":
                                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                                    model = (
                                        DecisionTreeClassifier(max_depth=hp["max_depth"], min_samples_split=hp["min_samples_split"],
                                                                min_samples_leaf=hp["min_samples_leaf"], random_state=42)
                                        if task_type == "Classification"
                                        else DecisionTreeRegressor(max_depth=hp["max_depth"], min_samples_split=hp["min_samples_split"],
                                                                    min_samples_leaf=hp["min_samples_leaf"], random_state=42)
                                    )

                                elif model_name == "XGBoost":
                                    # FIX: use_label_encoder was removed in xgboost>=2.0 —
                                    # passing it now raises a TypeError and would kill
                                    # training. Dropped it here.
                                    model = (
                                        XGBClassifier(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", 6),
                                            learning_rate=hp.get("learning_rate", 0.1), subsample=hp.get("subsample", 1.0),
                                            random_state=42, n_jobs=-1, eval_metric="logloss",
                                        )
                                        if task_type == "Classification"
                                        else XGBRegressor(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", 6),
                                            learning_rate=hp.get("learning_rate", 0.1), subsample=hp.get("subsample", 1.0),
                                            random_state=42, n_jobs=-1,
                                        )
                                    )

                                elif model_name == "LightGBM":
                                    model = (
                                        LGBMClassifier(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", -1),
                                            learning_rate=hp.get("learning_rate", 0.1), num_leaves=hp.get("num_leaves", 31),
                                            random_state=42, n_jobs=-1, verbose=-1,
                                        )
                                        if task_type == "Classification"
                                        else LGBMRegressor(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", -1),
                                            learning_rate=hp.get("learning_rate", 0.1), num_leaves=hp.get("num_leaves", 31),
                                            random_state=42, n_jobs=-1, verbose=-1,
                                        )
                                    )

                                else:  # Random Forest
                                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                                    model = (
                                        RandomForestClassifier(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", 7),
                                            min_samples_split=hp.get("min_samples_split", 5), min_samples_leaf=hp.get("min_samples_leaf", 2),
                                            random_state=42, n_jobs=-1,
                                        )
                                        if task_type == "Classification"
                                        else RandomForestRegressor(
                                            n_estimators=hp.get("n_estimators", 150), max_depth=hp.get("max_depth", 7),
                                            min_samples_split=hp.get("min_samples_split", 5), min_samples_leaf=hp.get("min_samples_leaf", 2),
                                            random_state=42, n_jobs=-1,
                                        )
                                    )

                                try:
                                    model.fit(X_train, y_train)
                                    preds = model.predict(X_test)

                                    cv_folds   = min(5, len(X_train))
                                    cv_scoring = "accuracy" if task_type == "Classification" else "r2"
                                    cv_scores  = cross_val_score(
                                        model, X_train, y_train, cv=cv_folds, scoring=cv_scoring, n_jobs=-1
                                    ) if cv_folds >= 2 else np.array([])

                                    elapsed = time.time() - start
                                    train_failed = False
                                except Exception as e:
                                    train_failed = True
                                    banner("err", f"Training failed: {str(e)}")

                            if not train_failed:
                                st.session_state.update({
                                    "trained_model"      : model,
                                    "trained_preds"      : preds.tolist(),
                                    "trained_y_test"     : y_test.tolist(),
                                    "trained_task"       : task_type,
                                    "trained_model_name" : model_name,
                                    "trained_target"     : target,
                                    "trained_features"   : X.columns.tolist(),
                                    "trained_pca_n"      : pca_n_after,
                                    "trained_hp"         : hp,
                                    "train_time"         : elapsed,
                                    "train_cv_scores"    : cv_scores.tolist(),
                                    "train_cv_metric"    : cv_scoring,
                                    "train_cv_folds"     : cv_folds,
                                    "scaler"             : scaler,
                                    "pca"                : pca,
                                })

                    # ── 6. Results ──
                    if "trained_model" in st.session_state:
                        preds      = np.array(st.session_state["trained_preds"])
                        y_test_arr = np.array(st.session_state["trained_y_test"])
                        task       = st.session_state["trained_task"]
                        mdl_name   = st.session_state["trained_model_name"]
                        pca_n      = st.session_state["trained_pca_n"]
                        used_hp    = st.session_state.get("trained_hp", {})
                        cv_scores  = np.array(st.session_state.get("train_cv_scores", []))
                        cv_metric  = st.session_state.get("train_cv_metric", "")
                        cv_folds   = st.session_state.get("train_cv_folds", 5)

                        divider()
                        eyebrow(f"Results — {mdl_name}")

                        if pca_n:
                            banner("warn", f"PCA applied → reduced to <strong>{pca_n}</strong> components (90% variance retained).")

                        if used_hp:
                            hp_str = " &nbsp;·&nbsp; ".join(
                                f'<span style="color:#f0f6fc;">{k}</span>: <span style="color:#388bfd;">{v}</span>'
                                for k, v in used_hp.items()
                            )
                            st.markdown(
                                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.68rem;color:#8b949e;margin-bottom:0.8rem;">Params — {hp_str}</div>',
                                unsafe_allow_html=True,
                            )

                        # ── Metrics ──
                        if task == "Regression":
                            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                            r2   = r2_score(y_test_arr, preds)
                            mse  = mean_squared_error(y_test_arr, preds)
                            rmse = np.sqrt(mse)
                            mae  = mean_absolute_error(y_test_arr, preds)

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("R² Score", round(r2, 4))
                            m2.metric("MAE", round(mae, 4))
                            m3.metric("MSE", round(mse, 4))
                            m4.metric("RMSE", round(rmse, 4))

                            divider()
                            eyebrow("Actual vs Predicted")
                            sample_n = min(200, len(y_test_arr))
                            idx      = np.random.choice(len(y_test_arr), sample_n, replace=False)
                            fig, ax  = plt.subplots(figsize=(7, 3.5))
                            fig.patch.set_facecolor("#0d1117")
                            ax.set_facecolor("#0d1117")
                            ax.scatter(y_test_arr[idx], preds[idx], color="#388bfd", alpha=0.55, s=18, edgecolors="none")
                            lims = [min(y_test_arr.min(), preds.min()), max(y_test_arr.max(), preds.max())]
                            ax.plot(lims, lims, color="#f78166", linewidth=1.2, linestyle="--", label="Perfect fit")
                            ax.set_xlabel("Actual", color="#8b949e", fontsize=8)
                            ax.set_ylabel("Predicted", color="#8b949e", fontsize=8)
                            ax.tick_params(colors="#8b949e", labelsize=7)
                            ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117", edgecolor="#21262d")
                            for s in ax.spines.values(): s.set_visible(False)
                            ax.grid(True, color="#161b22", linewidth=0.5)
                            plt.tight_layout()
                            st.pyplot(fig)
                            download_chart(fig, key="train_avp_download")
                            plt.close(fig)

                        else:
                            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
                            acc = accuracy_score(y_test_arr, preds)
                            st.metric("Test Accuracy", f"{round(acc * 100, 2)}%")

                            divider()
                            eyebrow("Classification Report")
                            report    = classification_report(y_test_arr, preds, output_dict=True, zero_division=0)
                            report_df = pd.DataFrame(report).transpose().round(3)
                            row_h     = 35
                            tbl_h     = min(500, 38 + len(report_df) * row_h)
                            st.dataframe(report_df, use_container_width=True, height=tbl_h)

                            divider()
                            eyebrow("Confusion Matrix")
                            cm       = confusion_matrix(y_test_arr, preds)
                            n_cls    = cm.shape[0]
                            fig_size = max(3, min(7, n_cls * 1.2))
                            fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
                            fig.patch.set_facecolor("#0d1117")
                            ax.set_facecolor("#0d1117")
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                        linewidths=0.5, linecolor="#161b22", annot_kws={"size": 9})
                            ax.set_xlabel("Predicted", color="#8b949e", fontsize=8)
                            ax.set_ylabel("Actual", color="#8b949e", fontsize=8)
                            ax.tick_params(colors="#8b949e", labelsize=7)
                            for s in ax.spines.values(): s.set_visible(False)
                            plt.tight_layout()
                            _, mid, _ = st.columns([1, 2, 1])
                            with mid:
                                st.pyplot(fig)
                                download_chart(fig, key="train_cm_download")
                            plt.close(fig)

                        # ── Cross-Validation ──
                        if len(cv_scores) > 0:
                            divider()
                            eyebrow(f"{cv_folds}-Fold Cross-Validation")
                            cv_label = "Accuracy" if cv_metric == "accuracy" else "R²"

                            cv1, cv2, cv3 = st.columns(3)
                            cv1.metric(f"Mean {cv_label}", round(cv_scores.mean(), 4))
                            cv2.metric("Std Dev", round(cv_scores.std(), 4))
                            cv3.metric("Min / Max", f"{round(cv_scores.min(),3)} / {round(cv_scores.max(),3)}")

                            fig, ax = plt.subplots(figsize=(5, 2.2))
                            fig.patch.set_facecolor("#0d1117")
                            ax.set_facecolor("#0d1117")
                            fold_colors = ["#388bfd" if s >= cv_scores.mean() else "#21262d" for s in cv_scores]
                            ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores, color=fold_colors, edgecolor="none")
                            ax.axhline(cv_scores.mean(), color="#f78166", linewidth=1.2, linestyle="--", label="Mean")
                            ax.set_ylabel(cv_label, color="#8b949e", fontsize=7)
                            ax.tick_params(colors="#8b949e", labelsize=7)
                            ax.legend(fontsize=7, labelcolor="#8b949e", facecolor="#0d1117", edgecolor="#21262d")
                            for s in ax.spines.values(): s.set_visible(False)
                            ax.yaxis.grid(True, color="#161b22", linewidth=0.5)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        divider()
                        banner("ok", f"<strong>{mdl_name}</strong> trained successfully &nbsp;·&nbsp; ⏱ {round(st.session_state['train_time'], 2)}s")

                        # ── Download (model only) ──
                        divider()
                        eyebrow("Download Model")

                        model_bundle = {
                            "model"   : st.session_state["trained_model"],
                            "scaler"  : st.session_state.get("scaler"),
                            "pca"     : st.session_state.get("pca"),
                            "features": st.session_state["trained_features"],
                            "target"  : st.session_state["trained_target"],
                            "task"    : st.session_state["trained_task"],
                            "params"  : st.session_state.get("trained_hp", {}),
                        }
                        buf = io.BytesIO()
                        pickle.dump(model_bundle, buf)
                        buf.seek(0)

                        st.download_button(
                            "⬇ Download Model (.pkl)",
                            data=buf,
                            file_name=f"{mdl_name.replace(' ', '_')}_model.pkl",
                            mime="application/octet-stream",
                            key="train_dl_model",
                            use_container_width=True,
                        )
