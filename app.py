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
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), engine="pyarrow")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        )
    return df
 
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
    padding: 2.8rem 0 2rem 0;
    border-bottom: 1px solid #161b22;
    margin-bottom: 2rem;
    position: relative;
}
 
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #30363d;
    margin-bottom: 0.6rem;
}
 
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    color: #f0f6fc;
    margin-bottom: 0.5rem;
}
 
.hero-title .dot {
    color: #f78166;
}
 
.hero-title .dim {
    color: #3d444d;
}
 
.hero-desc {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #3d444d;
    letter-spacing: 0.08em;
}
 
.hero-desc span {
    color: #388bfd;
}
 
/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
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
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #388bfd);
}
 
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #3d444d;
    margin-bottom: 0.45rem;
}
 
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
    color: #f0f6fc;
}
 
.stat-value.blue   { color: #388bfd; --accent: #388bfd; }
.stat-value.green  { color: #3fb950; --accent: #3fb950; }
.stat-value.yellow { color: #d29922; --accent: #d29922; }
.stat-value.red    { color: #f85149; --accent: #f85149; }
 
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
    color: #3d444d;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    padding: 0.4rem 0.9rem;
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
 
/* ── Subheaders ── */
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
 
# ================= HERO =================
st.markdown("""
<div class="hero">
    <div class="hero-title">DataPilot <span class="dim">Studio</span></div>
    <div class="hero-desc">
        Upload CSV <span>→</span> Explore Data <span>→</span> Generate Insights
    </div>
</div>
""", unsafe_allow_html=True)
 
# ================= FILE UPLOAD =================
file = st.file_uploader(
    "Upload CSV file (Max size: 100MB)",
    type=["csv"],
    label_visibility="collapsed"
)
 
if file is not None:
    if file.size > 100 * 1024 * 1024:
        st.markdown('<div class="banner banner-err">✕ &nbsp;File too large — upload under <strong>100MB</strong>.</div>', unsafe_allow_html=True)
        st.stop()
 
    if "file_name" not in st.session_state or st.session_state.file_name != file.name:
        st.session_state.file_name    = file.name
        st.session_state.original_df  = load_data(file.read())   
        st.session_state.df           = st.session_state.original_df.copy()
 
    if "df" not in st.session_state:
        st.markdown('<div class="banner banner-warn">⚠ Data not initialized — please re-upload.</div>', unsafe_allow_html=True)
        st.stop()
 
    df = st.session_state.df
 
    size_mb = file.size / (1024 * 1024)
    st.markdown(f'<div class="banner banner-ok">✓ &nbsp;<strong>{file.name}</strong> loaded &nbsp;·&nbsp; {size_mb:.2f} MB</div>', unsafe_allow_html=True)
 
    if len(df) > 200000:
        st.markdown('<div class="banner banner-warn">⚠ Large dataset — some operations may be slow.</div>', unsafe_allow_html=True)
 
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
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Column Names:")
            st.write(df.columns.tolist())
        with col2:
            st.subheader("Data Types:")
            st.write(df.dtypes)

        # ── Statistical Summary ──
        st.subheader("Statistical Summary")
        numeric_cols = df.select_dtypes(include=['number']).columns
        object_cols  = df.select_dtypes(include=['object', 'category']).columns
        bool_cols    = df.select_dtypes(include=['bool']).columns

        if len(numeric_cols) > 0:
            st.write("**Numerical Columns**")
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        if len(object_cols) > 0:
            st.write("**Categorical Columns**")
            st.dataframe(df[object_cols].describe(), use_container_width=True)
        if len(bool_cols) > 0:
            st.write("**Boolean Columns**")
            st.dataframe(df[bool_cols].describe(), use_container_width=True)

        # ── Missing Values ──
        st.subheader("Missing Values")
        missing         = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_df      = pd.DataFrame({
            "Missing Values": missing,
            "Percentage (%)": missing_percent.round(2)
        })
        missing_filtered = missing_df[missing_df["Missing Values"] > 0]
        if missing_filtered.empty:
            st.success("No missing values found 🎉")
        else:
            st.dataframe(missing_filtered, use_container_width=True)

        # ── DATA HEALTH SCORE ──
        st.subheader("Data Health Score")

        total_cells    = df.shape[0] * df.shape[1]
        missing_pct    = (df.isnull().sum().sum() / total_cells) * 100
        dup_pct        = (df.duplicated().sum() / len(df)) * 100

        outlier_counts = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            out    = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            outlier_counts.append(out)
        outlier_total = sum(outlier_counts)
        outlier_pct   = (outlier_total / len(df)) * 100 if len(df) > 0 else 0

        # Score calculation (100 mein se)
        missing_score = max(0, 100 - missing_pct * 2)
        dup_score     = max(0, 100 - dup_pct * 3)
        outlier_score = max(0, 100 - outlier_pct * 1.5)
        health_score  = round((missing_score * 0.4 + dup_score * 0.3 + outlier_score * 0.3), 1)

        if health_score >= 80:
            score_color = "#3fb950"
            score_label = "Excellent"
        elif health_score >= 60:
            score_color = "#d29922"
            score_label = "Fair"
        else:
            score_color = "#f85149"
            score_label = "Poor"

        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #21262d; border-radius:10px; padding:1.2rem; margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#3d444d; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.5rem;">Overall Health Score</div>
            <div style="font-size:2.8rem; font-weight:800; color:{score_color}; line-height:1;">{health_score}<span style="font-size:1rem; color:#3d444d;">/100</span></div>
            <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:{score_color}; margin-top:0.3rem;">{score_label}</div>
            <div style="margin-top:0.8rem; background:#161b22; border-radius:6px; height:6px; overflow:hidden;">
                <div style="width:{health_score}%; height:100%; background:{score_color}; border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Missing Score",  f"{round(missing_score, 1)}/100", f"{missing_pct:.1f}% missing")
        sc2.metric("Duplicate Score", f"{round(dup_score, 1)}/100",   f"{dup_pct:.1f}% duplicates")
        sc3.metric("Outlier Score",  f"{round(outlier_score, 1)}/100", f"{outlier_pct:.1f}% outliers")

        # ── OUTLIER DETECTION ──
        st.subheader("Outlier Detection (IQR Method)")

        if len(numeric_cols) == 0:
            st.info("No numeric columns found for outlier detection.")
        else:
            outlier_rows = []
            for col in numeric_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr    = q3 - q1
                lower  = q1 - 1.5 * iqr
                upper  = q3 + 1.5 * iqr
                count  = int(((df[col] < lower) | (df[col] > upper)).sum())
                pct    = round((count / len(df)) * 100, 2)
                outlier_rows.append({
                    "Column"         : col,
                    "Lower Bound"    : round(lower, 3),
                    "Upper Bound"    : round(upper, 3),
                    "Outlier Count"  : count,
                    "Outlier %"      : pct
                })

            outlier_df = pd.DataFrame(outlier_rows).sort_values("Outlier Count", ascending=False)
            st.dataframe(outlier_df, use_container_width=True)


# ---------- COLUMN ANALYZER ----------
    with eda_tab2:
        column = st.selectbox("Select Column", df.columns, key="col_analyzer")

        missing_count   = df[column].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_values   = df[column].nunique()

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Data Type",      str(df[column].dtype))
        colB.metric("Missing Values", int(missing_count))
        colC.metric("Missing %",      f"{missing_percent:.2f}%")
        colD.metric("Unique Values",  int(unique_values))

        # ── NUMERIC ──
        if np.issubdtype(df[column].dtype, np.number):

            mean_val   = df[column].mean()
            median_val = df[column].median()
            std_val    = df[column].std()
            min_val    = df[column].min()
            max_val    = df[column].max()
            skew_val   = df[column].skew()
            kurt_val   = df[column].kurt()

            colE, colF, colG = st.columns(3)
            colE.metric("Mean",     round(mean_val, 2))
            colF.metric("Median",   round(median_val, 2))
            colG.metric("Std Dev",  round(std_val, 2))

            colH, colI, colJ = st.columns(3)
            colH.metric("Min",      round(min_val, 2))
            colI.metric("Max",      round(max_val, 2))
            colJ.metric("Skewness", round(skew_val, 2))

            # ── Skewness & Kurtosis Interpretation ──
            st.markdown("#### Distribution Interpretation")

            if skew_val > 1:
                skew_label = "Highly Right Skewed"
                skew_color = "#f85149"
            elif skew_val > 0.5:
                skew_label = "Moderately Right Skewed"
                skew_color = "#d29922"
            elif skew_val < -1:
                skew_label = "Highly Left Skewed"
                skew_color = "#f85149"
            elif skew_val < -0.5:
                skew_label = "Moderately Left Skewed"
                skew_color = "#d29922"
            else:
                skew_label = "Approximately Symmetric"
                skew_color = "#3fb950"

            if kurt_val > 3:
                kurt_label = "Leptokurtic — heavy tails, sharp peak (outliers likely)"
                kurt_color = "#f85149"
            elif kurt_val < -1:
                kurt_label = "Platykurtic — light tails, flat distribution"
                kurt_color = "#d29922"
            else:
                kurt_label = "Mesokurtic — normal-like tails"
                kurt_color = "#3fb950"

            interp_l, interp_r = st.columns(2)
            interp_l.markdown(f"""
            <div style="background:#0d1117; border:1px solid #21262d; border-left:3px solid {skew_color};
                        border-radius:8px; padding:0.8rem 1rem;">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#3d444d;
                            text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.4rem;">
                    Skewness · {round(skew_val, 3)}
                </div>
                <div style="font-size:0.78rem; color:{skew_color};">{skew_label}</div>
            </div>
            """, unsafe_allow_html=True)

            interp_r.markdown(f"""
            <div style="background:#0d1117; border:1px solid #21262d; border-left:3px solid {kurt_color};
                        border-radius:8px; padding:0.8rem 1rem;">
                <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#3d444d;
                            text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.4rem;">
                    Kurtosis · {round(kurt_val, 3)}
                </div>
                <div style="font-size:0.78rem; color:{kurt_color};">{kurt_label}</div>
            </div>
            """, unsafe_allow_html=True)

            # sample for performance
            clean = df[column].dropna()
            if len(clean) > 5000:
                clean = clean.sample(5000, random_state=42)

            chart_l, chart_r = st.columns(2)

            # Histogram + KDE
            with chart_l:
                st.markdown("##### Distribution")
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                ax.hist(clean, bins=30, color="#388bfd", alpha=0.7, edgecolor="none")
                ax2 = ax.twinx()
                clean.plot.kde(ax=ax2, color="#f78166", linewidth=1.5)
                ax2.set_ylabel("")
                ax2.tick_params(left=False, right=False, labelleft=False, labelright=False)
                ax2.set_facecolor("#0d1117")
                for s in ax2.spines.values(): s.set_visible(False)
                ax.axvline(mean_val,   color="#d29922", linewidth=1.2, linestyle="--", label="Mean")
                ax.axvline(median_val, color="#3fb950", linewidth=1.2, linestyle="--", label="Median")
                ax.set_xlabel(column, color="#8b949e", fontsize=8)
                ax.set_ylabel("Count",  color="#8b949e", fontsize=8)
                ax.tick_params(colors="#8b949e", labelsize=7)
                for s in ax.spines.values(): s.set_visible(False)
                ax.legend(fontsize=7, labelcolor="#8b949e",
                          facecolor="#0d1117", edgecolor="#21262d")
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="hist_download")
                plt.close()

            # Box plot
            with chart_r:
                st.markdown("##### Box Plot")
                fig, ax = plt.subplots(figsize=(5, 3))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                ax.boxplot(clean, vert=False, patch_artist=True, widths=0.5,
                           boxprops    =dict(facecolor="#161b22", color="#388bfd"),
                           medianprops =dict(color="#f78166", linewidth=2),
                           whiskerprops=dict(color="#388bfd"),
                           capprops    =dict(color="#388bfd"),
                           flierprops  =dict(marker="o", color="#d29922",
                                             markersize=3, alpha=0.5))
                ax.set_xlabel(column, color="#8b949e", fontsize=8)
                ax.tick_params(colors="#8b949e", labelsize=7)
                ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="box_download")
                plt.close()

            # ── OUTLIER ROWS PREVIEW ──
            st.markdown("#### Outlier Rows Preview (IQR Method)")

            q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
            iqr    = q3 - q1
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr

            outlier_mask = (df[column] < lower) | (df[column] > upper)
            outlier_rows = df[outlier_mask]

            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Lower Bound", round(lower, 3))
            oc2.metric("Upper Bound", round(upper, 3))
            oc3.metric("Outlier Rows", len(outlier_rows))

            if outlier_rows.empty:
                st.success("No outliers found in this column 🎉")
            else:
                with st.expander(f"Preview {min(50, len(outlier_rows))} Outlier Rows"):
                    st.dataframe(outlier_rows.head(50), use_container_width=True)
                    if len(outlier_rows) > 50:
                        st.caption(f"Showing 50 of {len(outlier_rows)} outlier rows.")

        # ── CATEGORICAL ──
        else:
            vc = df[column].value_counts()

            mode_val = df[column].mode()
            if not mode_val.empty:
                most_frequent = mode_val[0]
                top_count     = vc.iloc[0]
            else:
                most_frequent = "N/A"
                top_count     = 0

            colE, colF = st.columns(2)
            colE.metric("Most Frequent Value", str(most_frequent))
            colF.metric("Top Value Count",     int(top_count))

            # ── Value Distribution % Table ──
            st.markdown("#### Value Distribution")
            vc_table = pd.DataFrame({
                "Value"      : vc.index,
                "Count"      : vc.values,
                "Percentage" : (vc.values / len(df) * 100).round(2)
            })
            vc_table["Percentage"] = vc_table["Percentage"].astype(str) + " %"
            st.dataframe(vc_table, use_container_width=True)

            MAX_BARS  = 20
            vc_plot   = vc.head(MAX_BARS)
            truncated = len(vc) > MAX_BARS

            chart_l, chart_r = st.columns(2)

            # Bar chart
            with chart_l:
                title = f"##### Top {MAX_BARS} Values" if truncated else "##### Value Counts"
                st.markdown(title)
                fig, ax = plt.subplots(figsize=(5, max(3, len(vc_plot) * 0.35)))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                colors = ["#388bfd" if i == 0 else "#161b22"
                          for i in range(len(vc_plot))]
                ax.barh(vc_plot.index[::-1], vc_plot.values[::-1],
                        color=colors[::-1], edgecolor="none")
                ax.set_xlabel("Count", color="#8b949e", fontsize=8)
                ax.tick_params(colors="#8b949e", labelsize=7)
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.grid(True, color="#161b22", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="bar_download")
                plt.close()

            # Pie chart
            with chart_r:
                if len(vc) <= 10:
                    st.markdown("##### Distribution")
                    pie_data = vc
                else:
                    st.markdown("##### Top 10 Share")
                    top10    = vc.head(10)
                    other    = vc.iloc[10:].sum()
                    pie_data = pd.concat([top10, pd.Series({"Other": other})])

                pie_colors = ["#388bfd","#58a6ff","#3fb950","#d29922",
                              "#f78166","#bc8cff","#79c0ff","#56d364",
                              "#e3b341","#ff7b72","#8b949e"]

                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#0d1117")
                wedges, texts, autotexts = ax.pie(
                    pie_data.values,
                    labels=pie_data.index,
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
                download_chart(fig, key="pie_download")
                plt.close()


# ---------- CORRELATION ----------
    with eda_tab3:
        numeric_df = df.select_dtypes(include=np.number)
        num_cols   = numeric_df.shape[1]
 
        if num_cols < 2:
            st.warning("⚠ Not enough numeric columns for correlation.")
        elif num_cols > 25:
            st.warning(f"⚠ Too many numeric columns ({num_cols}). Skipping heatmap for performance.")
            st.info("Tip: Remove columns or use Data Cleaning tab to reduce features.")
        else:
            corr = numeric_df.corr()
 
            # ── Heatmap ──
            st.markdown("##### Correlation Heatmap")
            annot = num_cols <= 15 
 
            fig4, ax4 = plt.subplots(figsize=(max(6, num_cols * 0.6), max(5, num_cols * 0.5)))
            fig4.patch.set_facecolor("#0d1117")
            ax4.set_facecolor("#0d1117")
 
            sns.heatmap(
                corr,
                annot=annot,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax4,
                linewidths=0.4,
                linecolor="#161b22",
                annot_kws={"size": 7, "color": "#f0f6fc"},
                cbar_kws={"shrink": 0.8}
            )
 
            ax4.tick_params(colors="#8b949e", labelsize=7)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
            ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)
 
            # colorbar styling
            cbar = ax4.collections[0].colorbar
            cbar.ax.tick_params(colors="#8b949e", labelsize=7)
            cbar.ax.yaxis.label.set_color("#8b949e")
 
            for s in ax4.spines.values(): s.set_visible(False)

            plt.tight_layout()
            st.pyplot(fig4)
            download_chart(fig4, "corr_heatmap_download")
            plt.close(fig4)
 
            # ── Top correlated pairs ──
            st.markdown("##### Top Correlated Pairs")
            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    .stack()
                    .reset_index()
            )
            corr_pairs.columns    = ["Feature 1", "Feature 2", "Correlation"]
            corr_pairs["Abs"]     = corr_pairs["Correlation"].abs()
            corr_pairs            = (corr_pairs
                                     .sort_values("Abs", ascending=False)
                                     .drop(columns="Abs")
                                     .head(10)
                                     .reset_index(drop=True))
            corr_pairs["Correlation"] = corr_pairs["Correlation"].round(4)
            st.dataframe(corr_pairs, use_container_width=True)


# ---------- VISUALIZATION ----------
    with eda_tab4:
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot",
             "Line Chart", "Violin Plot", "Pie Chart"],
            key="chart_type"
        )
 
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols     = df.select_dtypes(include="object").columns.tolist()
 
        x_col  = st.selectbox("Select X-axis", df.columns, key="x_col")
        y_col  = None
        hue_col = None
 
        if chart_type in ["Scatter Plot", "Line Chart"]:
            y_col = st.selectbox("Select Y-axis", df.columns, key="y_col")
 
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
            with st.spinner("Generating chart... ⏳"):
                try:
                    # ── sample large data ──
                    plot_df = df.sample(10000, random_state=42) if len(df) > 10000 else df
 
                    if chart_type == "Histogram":
                        if x_col not in numeric_cols:
                            st.error("❌ Histogram requires a numeric column.")
                        else:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.histplot(plot_df[x_col].dropna(), bins=30,
                                         color="#388bfd", alpha=0.8, ax=ax)
                            ax.set_xlabel(x_col)
                            ax.set_ylabel("Count")
                            style_ax(fig, ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            download_chart(fig, "histogram_download")
                            plt.close(fig)
 
                    elif chart_type == "Box Plot":
                        if x_col not in numeric_cols:
                            st.error("❌ Box Plot requires a numeric column.")
                        else:
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
                            download_chart(fig, "box_download")
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
                        download_chart(fig, "bar_download")
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
                                download_chart(fig, "scatter_download")
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
                            download_chart(fig, "line_download")
                            plt.close(fig)
 
                    elif chart_type == "Violin Plot":
                        if x_col not in numeric_cols:
                            st.error("❌ Violin Plot requires a numeric column.")
                        else:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            sns.violinplot(x=plot_df[x_col].dropna(), ax=ax,
                                           color="#388bfd", linecolor="#161b22",
                                           linewidth=0.8, inner="box")
                            ax.set_xlabel(x_col)
                            style_ax(fig, ax)
                            plt.tight_layout()
                            st.pyplot(fig)
                            download_chart(fig, "violin_download")
                            plt.close(fig)
 
                    elif chart_type == "Pie Chart":
                        counts = plot_df[x_col].value_counts().head(8)
                        if len(counts) > 10:
                            st.warning("⚠ Too many unique values for Pie Chart. Showing top 8.")
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
                        download_chart(fig, "pie_download")
                        plt.close(fig)
 
                except Exception as e:
                    st.error(f"❌ Error generating chart: {e}")


# ---------- Missing Values ----------
    with eda_tab5:
        st.subheader("Missing Values Table")

        df = st.session_state.df

        # -------- Missing calculation --------
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

            # -------- Fill section --------
            st.markdown("### Fill Missing Values")
            col1, col2 = st.columns(2)

            with col1:
                selected_col = st.selectbox(
                    "Select Column",
                    missing_df["Column"].tolist(),
                    key="col_select"
                )
            with col2:
                method = st.selectbox(
                    "Select Method",
                    ["Mean", "Median", "Mode"],
                    key="method_select"
                )

            # -------- Fill button --------
            if st.button("Fill Missing", key="fill_btn"):
                if method in ["Mean", "Median"] and not np.issubdtype(df[selected_col].dtype, np.number):
                    st.error("Mean/Median can only be applied to numeric columns ❌")
                else:
                    if method == "Mean":
                        value = df[selected_col].mean()
                    elif method == "Median":
                        value = df[selected_col].median()
                    else:
                        mode_val = df[selected_col].mode()
                        value = mode_val[0] if not mode_val.empty else None

                    if value is None:
                        st.warning("No valid value found to fill ❌")
                    else:
                        df[selected_col] = df[selected_col].fillna(value)
                        st.session_state.df = df
                        st.session_state["last_action"] = f"'{selected_col}' filled using {method}"
                        st.rerun()

        # -------- Show success after rerun --------
        if "last_action" in st.session_state:
            st.success(st.session_state["last_action"] + " ✅")
            del st.session_state["last_action"]

        # -------- Delete Column --------
        st.markdown("### Delete a Column")
        df = st.session_state.df  
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

        # -------- Reset --------
        if st.button("🔄 Reset to Original Dataset", key="fill_reset_btn"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state["df_history"] = []
            st.session_state["last_action"] = "Dataset reset to original"
            st.rerun()

        # -------- Download cleaned data --------
        st.markdown("### Download Cleaned Dataset")
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

        # -------- Select Column to check duplicates --------
        st.markdown("### Select Column to Check Duplicates")

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

        # -------- Stats --------
        duplicate_count = int(df.duplicated(subset=subset).sum())
        total_rows = len(df)
        dup_percent = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Duplicate Rows", f"{duplicate_count:,}")
        col3.metric("Duplicate %", f"{dup_percent}%")

        # -------- Preview --------
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

        # -------- Message --------
        if "dup_msg" in st.session_state:
            st.success(st.session_state["dup_msg"] + " ✅")
            del st.session_state["dup_msg"]

        # -------- Actions --------
        st.markdown("### Actions")
        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button(
                "🗑 Delete All Duplicates",
                key="delete_dup",
                disabled=duplicate_count == 0
            ):
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
                st.session_state["dup_msg"] = "Dataset reset to original"
                st.rerun()

        # -------- Download --------
        st.markdown("### Download Dataset")
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
