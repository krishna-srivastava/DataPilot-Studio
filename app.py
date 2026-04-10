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
def load_data(file):
    df = pd.read_csv(file, engine="pyarrow")

    # 🔥 Safe conversion (only object columns)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        )
    return df

# ================= PAGE CONFIG =================
st.set_page_config(page_title="DataPilot Studio", layout="wide")

st.title("🚀 DataPilot Studio")
st.markdown("Upload CSV → Explore Data → Generate Insights")

# ================= FILE UPLOAD =================
file = st.file_uploader(
    "Upload CSV file (Max size: 100MB)",
    type=["csv"]
)

if file is not None:

    if file.size > 100 * 1024 * 1024:
        st.error("File too large! Please upload file under 100MB.")
        st.stop()

    # Detect new file upload
    if "file_name" not in st.session_state or st.session_state.file_name != file.name:

        st.session_state.file_name = file.name
        st.session_state.original_df = load_data(file)
        st.session_state.df = st.session_state.original_df.copy()

    # 🔥 FIX: check before use
    if "df" not in st.session_state:
        st.warning("Data not initialized properly. Please re-upload file.")
        st.stop()

    df = st.session_state.df

    if len(df) > 200000:
        st.warning("Large dataset detected. Some operations may take time.")

    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5, eda_tab6 = st.tabs(
    ["Overview", "Column Analyzer", "Correlation", "Visualization", "Data Cleaning", "Duplicate rows cleaner"]
    )

        # ---------- OVERVIEW ----------
    with eda_tab1:

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])
            st.write("Column Names:")
            st.write(df.columns.tolist())

        with col2:
            st.write("Data Types:")
            st.write(df.dtypes)

        st.subheader("Statistical Summary")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        missing_df = pd.DataFrame({
            "Missing Values": missing,
            "Percentage (%)": missing_percent.round(2)
        })

        st.dataframe(missing_df[missing_df["Missing Values"] > 0])

        st.subheader("Duplicate Rows")
        st.write("Total Duplicate Rows:", df.duplicated().sum())


        # ---------- COLUMN ANALYZER ----------
    with eda_tab2:

        column = st.selectbox("Select Column", df.columns)

        missing_count = df[column].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_values = df[column].nunique()

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Data Type", str(df[column].dtype))
        colB.metric("Missing Values", int(missing_count))
        colC.metric("Missing %", f"{missing_percent:.2f}%")
        colD.metric("Unique Values", int(unique_values))

        if np.issubdtype(df[column].dtype, np.number):

            mean_val = df[column].mean()
            median_val = df[column].median()
            std_val = df[column].std()
            min_val = df[column].min()
            max_val = df[column].max()
            skew_val = df[column].skew()

            colE, colF, colG = st.columns(3)
            colE.metric("Mean", round(mean_val, 2))
            colF.metric("Median", round(median_val, 2))
            colG.metric("Std Dev", round(std_val, 2))

            colH, colI, colJ = st.columns(3)
            colH.metric("Min", round(min_val, 2))
            colI.metric("Max", round(max_val, 2))
            colJ.metric("Skewness", round(skew_val, 2))

        else:
            mode_val = df[column].mode()
            if not mode_val.empty:
                most_frequent = mode_val[0]
                top_count = df[column].value_counts().iloc[0]
            else:
                most_frequent = "N/A"
                top_count = 0

            colE, colF = st.columns(2)

            # 🔥 FIX (important)
            colE.metric("Most Frequent Value", str(most_frequent))

            colF.metric("Top Value Count", int(top_count))

        # ---------- CORRELATION ----------
    with eda_tab3:

        numeric_df = df.select_dtypes(include=np.number)

        num_cols = numeric_df.shape[1]

        if num_cols < 2:
            st.warning("Not enough numeric columns.")

        elif num_cols > 25:
            st.warning(f"⚠ Too many numeric columns ({num_cols}). Skipping heatmap for performance reasons.")
            st.info("Tip: Reduce features or use sampling for large datasets.")

        else:
            corr = numeric_df.corr()
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
            st.pyplot(fig4)
            download_chart(fig4, "corr_heatmap_download") 
            plt.close(fig4)

    with eda_tab4:
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Histogram",
                "Box Plot",
                "Bar Chart",
                "Scatter Plot",
                "Line Chart",
                "Violin Plot",
                "Pie Chart"
            ]
        )

        numeric_cols = df.select_dtypes(include=np.number).columns

        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = None

        if chart_type in ["Scatter Plot", "Line Chart"]:
            y_col = st.selectbox("Select Y-axis", df.columns)

        show_chart = st.button("Show Chart")

        if show_chart:
            try:

                if chart_type == "Histogram":

                    if x_col not in numeric_cols:
                        st.error("Histogram requires a numeric column")
                    else:
                        with st.spinner("Generating chart... Please wait ⏳"):
                            fig, ax = plt.subplots()
                            sns.histplot(df[x_col].dropna(), bins=20, ax=ax)
                            st.pyplot(fig)
                            download_chart(fig, "histogram_download")
                            plt.close(fig)

                elif chart_type == "Box Plot":

                    if x_col not in numeric_cols:
                        st.error("Box Plot requires a numeric column")
                    else:
                        with st.spinner("Generating chart... Please wait ⏳"):
                            fig, ax = plt.subplots()
                            sns.boxplot(x=df[x_col].dropna(), ax=ax)
                            st.pyplot(fig)
                            download_chart(fig, "box_download")
                            plt.close(fig)

                elif chart_type == "Bar Chart":
                    with st.spinner("Generating chart... Please wait ⏳"):
                        counts = df[x_col].value_counts().head(10)

                        fig, ax = plt.subplots()
                        ax.bar(counts.index.astype(str), counts.values)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        download_chart(fig, "bar_download")
                        plt.close(fig)


                elif chart_type == "Scatter Plot":
                    if x_col not in numeric_cols or y_col not in numeric_cols:
                        st.error("Scatter Plot requires numeric columns")

                    else:
                        scatter_df = df[[x_col, y_col]].dropna()
                        if scatter_df.empty:
                            st.warning("No valid data available after removing missing values.")

                        else:
                            with st.spinner("Generating chart... Please wait ⏳"):
                                fig, ax = plt.subplots()
                                sns.scatterplot(
                                    x=scatter_df[x_col],
                                    y=scatter_df[y_col],
                                    ax=ax
                                )
                                st.pyplot(fig)
                                download_chart(fig, "scatter_download")
                                plt.close(fig)


                elif chart_type == "Line Chart":

                    if x_col not in numeric_cols or y_col not in numeric_cols:
                        st.error("Line Chart requires numeric columns")
                    else:
                        with st.spinner("Generating chart... Please wait ⏳"):
                            fig, ax = plt.subplots()
                            line_df = df[[x_col, y_col]].dropna()
                            sns.lineplot(x=line_df[x_col], y=line_df[y_col], ax=ax)
                            st.pyplot(fig)
                            download_chart(fig, "line_download")
                            plt.close(fig)

                elif chart_type == "Violin Plot":

                    if x_col not in numeric_cols:
                        st.error("Violin Plot requires a numeric column")
                    else:
                        with st.spinner("Generating chart... Please wait ⏳"):
                            fig, ax = plt.subplots()
                            sns.violinplot(x=df[x_col].dropna(), ax=ax)
                            st.pyplot(fig)
                            download_chart(fig, "violin_download")
                            plt.close(fig)

                elif chart_type == "Pie Chart":
                    with st.spinner("Generating chart... Please wait ⏳"):
                        counts = df[x_col].value_counts().head(6)
                        fig, ax = plt.subplots()
                        ax.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%')
                        ax.axis("equal")

                        fig.tight_layout()
                        st.pyplot(fig)        # Chart display
                        download_chart(fig, "pie_download")
                        plt.close(fig)

            except Exception as e:
                st.error(f"Error generating chart: {e}")

    with eda_tab5:
        st.subheader("Handle Missing Values")

        st.write("Columns with Missing Values")

        missing_summary = df.isnull().sum()
        if missing_summary.sum() == 0:
            st.success("No missing values found in dataset")
        else:
            st.dataframe(missing_summary[missing_summary > 0])

        missing_cols = df.columns[df.isnull().any()]
        if len(missing_cols) == 0:
            st.success("No columns with missing values")
        else:
            column = st.selectbox("Select Column", missing_cols)

        missing_count = df[column].isnull().sum()

        st.write(f"Missing values in column: {missing_count}")

        method = st.selectbox(
            "Fill Method",
            ["Mean", "Median", "Mode"]
        )

        missing_count = df[column].isnull().sum()

        st.write(f"Missing values in column: {missing_count}")

        # Buttons in same row
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fill Missing Values", disabled=bool(missing_count == 0)):
                if method == "Mean":

                    if np.issubdtype(df[column].dtype, np.number):
                        mean_val = round(df[column].mean(),2)
                        st.session_state.df[column] = df[column].fillna(mean_val)
                        st.success("Missing values filled with Mean")

                    else:
                        st.error("Mean can only be used for numeric columns")

                elif method == "Median":
                    if np.issubdtype(df[column].dtype, np.number):
                        st.session_state.df[column] = df[column].fillna(df[column].median())
                        st.success("Missing values filled with Median")
                    else:
                        st.error("Median can only be used for numeric columns")

                elif method == "Mode":
                    mode_val = df[column].mode()

                    if not mode_val.empty:
                        st.session_state.df[column] = df[column].fillna(mode_val[0])
                        st.success("Missing values filled with Mode")


        with col2:
            # Check if dataframe has missing values
            has_missing = st.session_state.df.isnull().sum().sum() > 0

            if st.button("Reset Dataset"):
                st.session_state.df = st.session_state.original_df.copy()
                st.success("Dataset reset to original")

        # ---------- DOWNLOAD ----------
        st.subheader("Download Cleaned Dataset")

        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Cleaned CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )


    with eda_tab6:
        st.subheader("Duplicate Rows Checker")

        df = st.session_state.df

        # Count duplicate rows
        duplicate_count = df.duplicated().sum()

        st.write(f"Total Duplicate Rows: {duplicate_count}")

        # Preview duplicates
        if duplicate_count > 0:
            duplicates = df[df.duplicated()]
            st.write("Duplicate Rows Preview")
            st.dataframe(duplicates)

        col1, col2, col3 = st.columns(3)

        # Delete duplicates
        with col1:
            if st.button("Delete All Duplicates", disabled=bool(duplicate_count == 0), key="delete_dup"):
                st.session_state.df = df.drop_duplicates()
                st.success("All duplicate rows removed")
                st.rerun()

        # Reset dataset
        with col2:
            if st.button("Reset Dataset", key="reset_dup"):
                st.session_state.df = st.session_state.original_df.copy()
                st.success("Dataset reset to original")
                st.rerun()

        # Download dataset
        with col3:
            csv = st.session_state.df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                key="download_dup"
            )         
else:
    st.info("Upload a CSV file to begin.")
