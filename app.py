import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

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

/* ── Signature scan-line — DataPilot's own motif (distinct from Horizon's
   split solid/dashed line): a traveling gradient bar suggesting a "pilot
   scanning across data", not a timeline like Horizon's ── */
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

/* ── Section headers (shared helper target) ── */
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

/* ── Stat cards — corner-notch accent instead of Horizon's split-line top
   border, so the family resemblance is only in spirit, not identical ── */
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

/* ── Pill-style segmented controls (for later use) ── */
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

/* ── Tabs ── */
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
    """Section header — mirrors the old inline Space Mono paragraph pattern,
    but as one reusable call instead of repeated markdown blocks."""
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub_html}', unsafe_allow_html=True)

def eyebrow(title):
    """Small uppercase mono label — matches the original section-header style
    used throughout Overview/Column Analyzer/etc, kept for lighter-weight headers."""
    st.markdown(f'<p class="section-eyebrow">{title}</p>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

def banner(kind, text):
    """kind: ok | warn | err | info"""
    icons = {"ok": "✓", "warn": "⚠", "err": "✕", "info": "ℹ"}
    st.markdown(f'<div class="banner banner-{kind}">{icons.get(kind, "•")} &nbsp;{text}</div>', unsafe_allow_html=True)

# ================= HERO =================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">CSV EXPLORATION &amp; INSIGHTS</div>
    <div class="hero-title">DataPilot <span class="dim">Studio</span></div>
    <div class="hero-desc">
        Upload CSV <span>→</span> Explore Data <span>→</span> Generate Insights
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

# ================= FILE UPLOAD =================
file = st.file_uploader(
    "Upload CSV file (Max size: 50MB)",
    type=["csv"],
    label_visibility="collapsed"
)

if file is not None:
    if file.size > 50 * 1024 * 1024:
        banner("err", "File too large — upload under <strong>50MB</strong>.")
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

    # ================= TABS =================
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5, eda_tab6 = st.tabs([
        "Overview", "Column Analyzer", "Correlation",
        "Visualization", "Data Cleaning", "Duplicate Rows"
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
                st.stop()

            if not selected_others:
                banner("warn", "Select at least one column to compare against the target.")
                st.stop()

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
                # ── Positive correlation → Blue ──
                if val > 0:
                    if abs_val >= 0.8:
                        bg = "#0c2340"
                        fg = "#60a5fa"
                    elif abs_val >= 0.5:
                        bg = "#0a1a2e"
                        fg = "#79c0ff"
                    elif abs_val >= 0.2:
                        bg = "#0d1826"
                        fg = "#5b9ee8"
                    else:
                        bg = "#0d1117"
                        fg = "#4a7fb5"

                # ── Negative correlation → Red ──
                elif val < 0:
                    if abs_val >= 0.8:
                        bg = "#3a0d16"
                        fg = "#ff6b7a"
                    elif abs_val >= 0.5:
                        bg = "#2b0b12"
                        fg = "#f78166"
                    elif abs_val >= 0.2:
                        bg = "#1d0d12"
                        fg = "#e56b6f"
                    else:
                        bg = "#0d1117"
                        fg = "#a85b63"

                # ── Exactly zero ──
                else:
                    bg = "#0d1117"
                    fg = "#4a5260"

                return (
                    f"background-color:{bg};"
                    f"color:{fg};"
                    f"font-weight:500;"
                )

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
                [
                    "#0b1f3a",   # strong negative → deep navy
                    "#174ea6",   # negative → blue
                    "#111827",   # zero → dark slate
                    "#2563eb",   # positive → blue
                    "#7dd3fc"    # strong positive → light cyan
                ],
                N=256
            )

            fig4, ax4 = plt.subplots(
                figsize=(max(6, num_cols * 0.6), max(5, num_cols * 0.5))
            )

            # ── Dark theme ──
            fig4.patch.set_facecolor("#0d1117")
            ax4.set_facecolor("#0d1117")
            sns.heatmap(corr,annot=False,cmap=horizon_cmap,center=0,vmin=-1,vmax=1,ax=ax4,linewidths=0.7,linecolor="#0d1117",square=True,cbar_kws={"shrink": 0.80,"pad": 0.03})

            # ── Custom correlation values ──
            if annot:
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        value = corr.iloc[i, j]

                        if abs(value) >= 0.55:
                            text_color = "#ffffff"
                        else:
                            text_color = "#c7d0d9"

                        ax4.text(j + 0.5,i + 0.5,f"{value:.2f}",ha="center",va="center",fontsize=8,fontweight="bold",color=text_color)

            # ── Axis labels ──
            ax4.tick_params(colors="#8b949e",labelsize=7,length=0)
            ax4.set_xticklabels(ax4.get_xticklabels(),rotation=45,ha="right")
            ax4.set_yticklabels(ax4.get_yticklabels(),rotation=0)

            # ── Highlight target column + row ──
            target_idx = list(corr.columns).index(target_col)
            target_color = "#60a5fa"
            ax4.add_patch(
                plt.Rectangle((target_idx, 0),1,num_cols,fill=False,edgecolor=target_color,linewidth=2.2
                )
            )
            ax4.add_patch(
                plt.Rectangle((0, target_idx),num_cols,1,fill=False,edgecolor=target_color,linewidth=2.2
                )
            )

            cbar = ax4.collections[0].colorbar
            cbar.ax.tick_params(colors="#8b949e",labelsize=7,length=0)
            cbar.outline.set_edgecolor("#263241")
            cbar.outline.set_linewidth(0.8)

            for spine in ax4.spines.values():
                spine.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig4)
            download_chart(
                fig4,
                "corr_heatmap_download"
            )
            plt.close(fig4)


# ---------- VISUALIZATION ----------
    with eda_tab4:

        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Chart Settings</p>', unsafe_allow_html=True)

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot",
             "Line Chart", "Violin Plot", "Pie Chart"],
            key="chart_type"
        )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = df.select_dtypes(include="object").columns.tolist()
        all_cols     = df.columns.tolist()

        if chart_type in ["Histogram", "Box Plot", "Violin Plot"]:
            x_options = numeric_cols if numeric_cols else all_cols
            st.caption("ℹ️ Only numeric columns shown for this chart type.")
        elif chart_type in ["Bar Chart", "Pie Chart"]:
            x_options = cat_cols if cat_cols else all_cols
            st.caption("ℹ️ Only categorical columns shown for this chart type.")
        else:
            x_options = all_cols

        x_col   = st.selectbox("Select X-axis", x_options, key="x_col")
        y_col   = None
        hue_col = None

        if chart_type in ["Scatter Plot", "Line Chart"]:
            y_options = numeric_cols if numeric_cols else all_cols
            y_col = st.selectbox("Select Y-axis", y_options, key="y_col")

        if chart_type == "Scatter Plot" and cat_cols:
            hue_options = ["None"] + cat_cols
            hue_sel     = st.selectbox("Color by (optional)", hue_options, key="hue_col")
            hue_col     = None if hue_sel == "None" else hue_sel

        if len(df) > 10000:
            st.caption(f"⚡ Large dataset ({len(df):,} rows) — chart will use 10,000 random samples.")

        show_chart = st.button("Show Chart", key="show_chart_btn")

        def style_ax(fig, ax):
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.set_xlabel(ax.get_xlabel(), color="#8b949e", fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), color="#8b949e", fontsize=8)
            for s in ax.spines.values(): s.set_visible(False)
            ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
            ax.yaxis.grid(True, color="#161b22", linewidth=0.5)

        if show_chart:
            st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Chart Output</p>', unsafe_allow_html=True)

            with st.spinner("Generating chart... ⏳"):
                try:
                    plot_df = df.sample(10000, random_state=42) if len(df) > 10000 else df

                    if chart_type == "Histogram":
                        fig, ax = plt.subplots(figsize=(7, 4))
                        sns.histplot(plot_df[x_col].dropna(), bins=30,
                                     color="#388bfd", alpha=0.8, ax=ax)
                        ax.set_xlabel(x_col)
                        ax.set_ylabel("Count")
                        style_ax(fig, ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, "viz_histogram_download")
                        plt.close(fig)

                    elif chart_type == "Box Plot":
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.boxplot(plot_df[x_col].dropna(), vert=False,
                                   patch_artist=True, widths=0.5,
                                   boxprops    =dict(facecolor="#161b22", color="#388bfd"),
                                   medianprops =dict(color="#f78166", linewidth=2),
                                   whiskerprops=dict(color="#388bfd"),
                                   capprops    =dict(color="#388bfd"),
                                   flierprops  =dict(marker="o", color="#d29922",
                                                     markersize=3, alpha=0.5))
                        ax.set_xlabel(x_col)
                        ax.set_yticks([])
                        style_ax(fig, ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, "viz_box_download")
                        plt.close(fig)

                    elif chart_type == "Bar Chart":
                        counts = plot_df[x_col].value_counts().head(15)
                        fig, ax = plt.subplots(figsize=(7, 4))
                        colors  = ["#388bfd" if i == 0 else "#161b22"
                                   for i in range(len(counts))]
                        ax.barh(counts.index.astype(str)[::-1],
                                counts.values[::-1],
                                color=colors[::-1], edgecolor="none")
                        ax.set_xlabel("Count")
                        ax.set_ylabel(x_col)
                        style_ax(fig, ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, "viz_bar_download")
                        plt.close(fig)

                    elif chart_type == "Scatter Plot":
                        if x_col not in numeric_cols or y_col not in numeric_cols:
                            st.error("❌ Scatter Plot requires numeric X and Y columns.")
                        else:
                            cols_needed = [x_col, y_col] + ([hue_col] if hue_col else [])
                            scatter_df  = plot_df[cols_needed].dropna()

                            if scatter_df.empty:
                                st.warning("⚠ No valid data after removing missing values.")
                            else:
                                fig, ax = plt.subplots(figsize=(7, 4))
                                if hue_col:
                                    categories = scatter_df[hue_col].unique()
                                    palette    = ["#388bfd","#3fb950","#f78166",
                                                  "#d29922","#bc8cff","#79c0ff"]
                                    for i, cat in enumerate(categories):
                                        mask = scatter_df[hue_col] == cat
                                        ax.scatter(scatter_df.loc[mask, x_col],
                                                   scatter_df.loc[mask, y_col],
                                                   color=palette[i % len(palette)],
                                                   label=str(cat), alpha=0.6, s=18,
                                                   edgecolors="none")
                                    ax.legend(fontsize=7, labelcolor="#8b949e",
                                              facecolor="#0d1117", edgecolor="#21262d")
                                else:
                                    ax.scatter(scatter_df[x_col], scatter_df[y_col],
                                               color="#388bfd", alpha=0.5, s=18,
                                               edgecolors="none")
                                ax.set_xlabel(x_col)
                                ax.set_ylabel(y_col)
                                style_ax(fig, ax)
                                plt.tight_layout()
                                st.pyplot(fig)
                                download_chart(fig, "viz_scatter_download")
                                plt.close(fig)

                    elif chart_type == "Line Chart":
                        if x_col not in numeric_cols or y_col not in numeric_cols:
                            st.error("❌ Line Chart requires numeric X and Y columns.")
                        else:
                            line_df = plot_df[[x_col, y_col]].dropna().sort_values(x_col)
                            fig, ax = plt.subplots(figsize=(7, 4))
                            ax.plot(line_df[x_col], line_df[y_col],
                                    color="#388bfd", linewidth=1.5)
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            style_ax(fig, ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            download_chart(fig, "viz_line_download")
                            plt.close(fig)

                    elif chart_type == "Violin Plot":
                        fig, ax = plt.subplots(figsize=(7, 4))
                        sns.violinplot(x=plot_df[x_col].dropna(), ax=ax,
                                       color="#388bfd", linecolor="#161b22",
                                       linewidth=0.8, inner="box")
                        ax.set_xlabel(x_col)
                        style_ax(fig, ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                        download_chart(fig, "viz_violin_download")
                        plt.close(fig)

                    elif chart_type == "Pie Chart":
                        counts = plot_df[x_col].value_counts().head(8)
                        pie_colors = ["#388bfd","#3fb950","#f78166","#d29922",
                                      "#bc8cff","#79c0ff","#56d364","#e3b341"]
                        fig, ax = plt.subplots(figsize=(5, 5))
                        fig.patch.set_facecolor("#0d1117")
                        wedges, texts, autotexts = ax.pie(
                            counts.values,
                            labels=counts.index.astype(str),
                            colors=pie_colors[:len(counts)],
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
                        download_chart(fig, "viz_pie_download")
                        plt.close(fig)

                except Exception as e:
                    st.error(f"❌ Error generating chart: {e}")



# ---------- Missing Values ----------
    with eda_tab5:
        df = st.session_state.df

        # ── Missing Values Table ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Missing Values Table</p>', unsafe_allow_html=True)

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values in dataset 🎉")
        else:
            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values
            }).sort_values(by="Missing Values", ascending=False)

            st.caption("Only columns with missing values are shown below")
            st.dataframe(missing_df, use_container_width=True)

            st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

            # ── Fill Missing Values ──
            st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Fill Missing Values</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox(
                    "Select Column",
                    missing_df["Column"].tolist(),
                    key="col_select"
                )

            is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
            if is_numeric:
                method_options = ["Mean", "Median", "Mode", "Custom Value"]
            else:
                method_options = ["Mode", "Custom Value"]

            with col2:
                method = st.selectbox(
                    "Select Method",
                    method_options,
                    key="method_select"
                )

            custom_value = None
            if method == "Custom Value":
                custom_value = st.text_input(
                    "Enter custom value to fill",
                    placeholder='e.g. 0 or "Unknown"',
                    key="custom_fill_val"
                )

            if st.button("Fill Missing", key="fill_btn"):
                if method == "Mean":
                    value = df[selected_col].mean()
                elif method == "Median":
                    value = df[selected_col].median()
                elif method == "Mode":
                    mode_val = df[selected_col].mode()
                    value = mode_val[0] if not mode_val.empty else None
                else:
                    if custom_value == "" or custom_value is None:
                        st.warning("Please enter a custom value ❌")
                        value = None
                    else:
                        if is_numeric:
                            try:
                                value = float(custom_value)
                            except ValueError:
                                st.error("Numeric column — please enter a number ❌")
                                value = None
                        else:
                            value = custom_value

                if value is not None:
                    if "df_history" not in st.session_state:
                        st.session_state["df_history"] = []
                    st.session_state["df_history"].append(st.session_state.df.copy())
                    df[selected_col] = df[selected_col].fillna(value)
                    st.session_state.df = df
                    st.session_state["last_action"] = f"'{selected_col}' filled using {method}"
                    st.rerun()

        # ── Success message ──
        if "last_action" in st.session_state:
            st.success(st.session_state["last_action"] + " ✅")
            del st.session_state["last_action"]

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Rename Column ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Rename a Column</p>', unsafe_allow_html=True)

        df = st.session_state.df

        if len(df.columns) == 0:
            st.warning("No columns remaining to rename. Reset the dataset first. ❌")
        else:
            rn1, rn2, rn3 = st.columns([2, 2, 1])
            with rn1:
                rename_col = st.selectbox(
                    "Select Column",
                    df.columns.tolist(),
                    key="rename_col"
                )
            with rn2:
                new_name = st.text_input(
                    "New Name",
                    placeholder="Enter new column name",
                    key="rename_new"
                )
            with rn3:
                st.markdown("<br>", unsafe_allow_html=True)
                rename_btn = st.button("Rename", key="rename_btn")

            if rename_btn:
                if not new_name.strip():
                    st.error("New name cannot be empty ❌")
                elif new_name.strip() in df.columns and new_name.strip() != rename_col:
                    st.error(f"Column '{new_name}' already exists ❌")
                else:
                    if "df_history" not in st.session_state:
                        st.session_state["df_history"] = []
                    st.session_state["df_history"].append(st.session_state.df.copy())
                    st.session_state.df = st.session_state.df.rename(columns={rename_col: new_name.strip()})
                    st.session_state["last_action"] = f"'{rename_col}' renamed to '{new_name.strip()}'"
                    st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Delete Column ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Delete a Column</p>', unsafe_allow_html=True)

        df = st.session_state.df

        if len(df.columns) == 0:
            st.warning("No columns remaining to delete. Reset the dataset first. ❌")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                del_col = st.selectbox(
                    "Select column to delete",
                    df.columns.tolist(),
                    key="fill_del_col"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Delete Column", key="fill_del_btn"):
                    if "df_history" not in st.session_state:
                        st.session_state["df_history"] = []
                    st.session_state["df_history"].append(st.session_state.df.copy())
                    st.session_state.df = st.session_state.df.drop(columns=[del_col])
                    st.session_state["last_action"] = f"'{del_col}' column deleted"
                    st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Undo / Reset ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Undo / Reset</p>', unsafe_allow_html=True)

        history = st.session_state.get("df_history", [])
        undo_col, reset_col = st.columns(2)

        with undo_col:
            if st.button(
                f"↩ Undo Last Action ({len(history)} steps)",
                key="undo_btn",
                disabled=len(history) == 0
            ):
                st.session_state.df = st.session_state["df_history"].pop()
                st.session_state["last_action"] = "Last action undone"
                st.rerun()

        with reset_col:
            if st.button("🔄 Reset to Original Dataset", key="fill_reset_btn"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state["df_history"] = []
                st.session_state["last_action"] = "Dataset reset to original"
                st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Download ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Download Cleaned Dataset</p>', unsafe_allow_html=True)
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )



# ---------- Duplicated Values ----------
    with eda_tab6:
        df = st.session_state.df

        # ── Check Mode ──
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

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Stats ──
        duplicate_count = int(df.duplicated(subset=subset).sum())
        total_rows      = len(df)
        dup_percent     = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows",     f"{total_rows:,}")
        col2.metric("Duplicate Rows", f"{duplicate_count:,}")
        col3.metric("Duplicate %",    f"{dup_percent}%")

        # ── Preview ──
        if duplicate_count == 0:
            st.success("No duplicate rows found 🎉")
        else:
            st.warning(f"⚠ {duplicate_count:,} duplicate rows found.")
            with st.expander("Preview Duplicate Rows"):
                st.dataframe(
                    df[df.duplicated(subset=subset)].head(50),
                    use_container_width=True
                )
                if duplicate_count > 50:
                    st.caption(f"Showing 50 of {duplicate_count:,} duplicate rows.")

        # ── Message ──
        if "dup_msg" in st.session_state:
            st.success(st.session_state["dup_msg"] + " ✅")
            del st.session_state["dup_msg"]

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Actions ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Actions</p>', unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button(
                "🗑 Delete All Duplicates",
                key="delete_dup",
                disabled=duplicate_count == 0
            ):
                if "df_history" not in st.session_state:
                    st.session_state["df_history"] = []
                st.session_state["df_history"].append(st.session_state.df.copy())
                before = len(st.session_state.df)
                st.session_state.df = st.session_state.df.drop_duplicates(
                    subset=subset
                ).reset_index(drop=True)
                after = len(st.session_state.df)
                st.session_state["dup_msg"] = f"{before - after:,} duplicate rows removed"
                st.rerun()

        with btn2:
            if st.button("🔄 Reset to Original Dataset", key="reset_dup"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state["df_history"] = []
                st.session_state["dup_msg"] = "Dataset reset to original"
                st.rerun()

        # ── Undo ──
        history = st.session_state.get("df_history", [])
        if st.button(
            f"↩ Undo Last Action ({len(history)} steps)",
            key="undo_dup_btn",
            disabled=len(history) == 0
        ):
            st.session_state.df = st.session_state["df_history"].pop()
            st.session_state["dup_msg"] = "Last action undone"
            st.rerun()

        st.markdown("<hr style='border:none; border-top:1px solid #161b22; margin:1.5rem 0;'>", unsafe_allow_html=True)

        # ── Download ──
        st.markdown('<p style="font-family:Space Mono,monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#8b949e; margin-bottom:0.8rem;">Download Dataset</p>', unsafe_allow_html=True)

        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            key="download_dup"
        )

       
else:
    st.info("Upload a CSV file to begin.")
