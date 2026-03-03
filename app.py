import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# ================= PAGE CONFIG =================
st.set_page_config(page_title="DataPilot Studio", layout="wide")

st.title("🚀 DataPilot Studio")
st.markdown("Upload CSV → Analyze → Train Model")

# ================= FILE UPLOAD =================
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    if file.size > 200 * 1024 * 1024:
        st.error("File too large! Please upload file under 200MB.")
        st.stop()

    df = pd.read_csv(file)

    if len(df) > 200000:
        st.warning("Large dataset detected. Some operations may take time.")

    # ================= MAIN TABS =================
    tab1, tab2 = st.tabs(["📊 EDA Section", "🤖 ML Section"])

    # ============================================================
    # ======================= EDA TAB =============================
    # ============================================================
    with tab1:

        eda_tab1, eda_tab2, eda_tab3 = st.tabs(
            ["Overview", "Column Analyzer", "Correlation"]
        )

        # ---------- OVERVIEW ----------
        with eda_tab1:

            st.subheader("Dataset Preview")
            st.dataframe(df.head())

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

            colA, colB, colC = st.columns(3)
            colA.metric("Data Type", str(df[column].dtype))
            colB.metric("Missing Values", int(df[column].isnull().sum()))
            colC.metric("Unique Values", int(df[column].nunique()))

            if np.issubdtype(df[column].dtype, np.number):

                colD, colE, colF = st.columns(3)
                colD.metric("Mean", round(df[column].mean(), 2))
                colE.metric("Median", round(df[column].median(), 2))
                colF.metric("Std Dev", round(df[column].std(), 2))

                fig, ax = plt.subplots()
                ax.hist(df[column].dropna(), bins=20)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                ax2.boxplot(df[column].dropna())
                st.pyplot(fig2)

            else:
                top_values = df[column].value_counts().head(10)
                st.dataframe(top_values)

                fig3, ax3 = plt.subplots()
                ax3.bar(top_values.index.astype(str), top_values.values)
                plt.xticks(rotation=45)
                st.pyplot(fig3)

        # ---------- CORRELATION ----------
        with eda_tab3:

            numeric_df = df.select_dtypes(include=np.number)

            num_cols = numeric_df.shape[1]

            if num_cols < 2:
                st.warning("Not enough numeric columns.")

            elif num_cols > 30:
                st.warning(f"⚠ Too many numeric columns ({num_cols}). Skipping heatmap for performance reasons.")
                st.info("Tip: Reduce features or use sampling for large datasets.")

            else:
                corr = numeric_df.corr()

                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
                st.pyplot(fig4)

    # ============================================================
    # ======================= ML TAB ==============================
    # ============================================================

    with tab2:
        st.subheader("🚀 ML Training Section")
        
        target = st.selectbox("Select Target Column", df.columns)

        if target:

            X = df.drop(columns=[target])
            y = df[target]

            missing_count = X.isnull().sum().sum()
            non_numeric_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

            dataset_valid = True
            # 🔴 Small Dataset Protection
            min_required_rows = 10

            if len(df) < min_required_rows:
                st.error(f"❌ Dataset must have at least {min_required_rows} rows for training.")
                dataset_valid = False

            if missing_count > 0:
                st.error(f"❌ Dataset contains {missing_count} missing values. Please clean your data.")
                dataset_valid = False

            elif len(non_numeric_cols) > 0:
                st.error(f"❌ Non-numeric columns detected: {non_numeric_cols}. Please encode them.")
                dataset_valid = False
            else:
                st.success("✅ Dataset is clean and ready for training.")


           # 🔴 Safe Problem Type Detection
            if y.dtype == "object" or str(y.dtype) == "category":
                problem_type = "Classification"

            elif np.issubdtype(y.dtype, np.number):
                if y.nunique() <= 10:
                    problem_type = "Classification"
                else:
                    problem_type = "Regression"

            else:
                st.error("❌ Unsupported target column type.")
                dataset_valid = False
                problem_type = None

            # Show only if detected
            if problem_type:
                st.success(f"Detected Problem Type: {problem_type}")

            # 🔴 Single Class Protection
            if problem_type == "Classification" and y.nunique() < 2:
                st.error("❌ Target column contains only ONE class. At least 2 classes are required for classification.")
                dataset_valid = False

            # 🔴 Regression Safety Double Check
            if problem_type == "Regression" and not np.issubdtype(y.dtype, np.number):
                st.error("❌ Regression requires numeric target column.")
                dataset_valid = False

            # Model selection
            if problem_type == "Classification":
                model_name = st.selectbox(
                    "Select Model",
                    ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
                )
            else:
                model_name = st.selectbox(
                    "Select Model",
                    ["Linear Regression", "KNN Regressor", "Decision Tree Regressor", "Random Forest Regressor", "SVR"]
                )

            if model_name not in ["Linear Regression"]:
                use_custom = st.checkbox("⚙️ Use Custom Hyperparameters")

            params = {}

            # ================= HYPERPARAMETERS =================

            if model_name == "Logistic Regression":
                if use_custom:
                    params["C"] = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
                    params["max_iter"] = st.slider("Max Iterations", 100, 5000, 1000)
                else:
                    params["C"] = 1.0
                    params["max_iter"] = 1000

            elif model_name == "KNN":
                if use_custom:
                    params["n_neighbors"] = st.slider("Number of Neighbors (K)", 1, 20, 5)
                else:
                    params["n_neighbors"] = 5

            elif model_name == "KNN Regressor":
                if use_custom:
                    params["n_neighbors"] = st.slider("Number of Neighbors (K)", 1, 20, 5)
                else:
                    params["n_neighbors"] = 5

            elif model_name == "SVM":
                if use_custom:
                    params["C"] = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
                    params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                else:
                    params["C"] = 1.0
                    params["kernel"] = "rbf"

            elif model_name == "SVR":
                if use_custom:
                    params["C"] = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
                    params["kernel"] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                else:
                    params["C"] = 1.0
                    params["kernel"] = "rbf" 

            elif model_name == "Random Forest":
                if use_custom:
                    params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100)
                    params["max_depth"] = st.slider("Max Depth", 1, 50, 10)
                    params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
                    params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1)
                else:
                    params["n_estimators"] = 100
                    params["max_depth"] = None
                    params["min_samples_split"] = 2
                    params["min_samples_leaf"] = 1

            elif model_name == "Decision Tree Regressor":
                if use_custom:
                    params["max_depth"] = st.slider("Max Depth", 1, 50, 5)
                    params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
                    params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1)
                else:
                    params["max_depth"] = None
                    params["min_samples_split"] = 2
                    params["min_samples_leaf"] = 1

            elif model_name == "Decision Tree":
                if use_custom:
                    params["max_depth"] = st.slider("Max Depth", 1, 50, 5)
                    params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
                    params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1)
                else:
                    params["max_depth"] = None
                    params["min_samples_split"] = 2
                    params["min_samples_leaf"] = 1

            elif model_name == "Random Forest Regressor":
                if use_custom:
                    params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100)
                    params["max_depth"] = st.slider("Max Depth", 1, 50, 10)
                    params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2)
                    params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1)
                else:
                    params["n_estimators"] = 100
                    params["max_depth"] = None
                    params["min_samples_split"] = 2
                    params["min_samples_leaf"] = 1

            # ================= TRAIN BUTTON =================

            # 🔥 AUTO SAMPLING (PLACE HERE)
            max_rows = 20000
            if len(X) > max_rows:
                st.warning(f"Large dataset detected ({len(X)} rows).")
                sample_size = min(max_rows, len(X))

                sample_indices = X.sample(sample_size, random_state=42).index
                X = X.loc[sample_indices]
                y = y.loc[sample_indices]
                st.info(f"Training will use {sample_size} sampled rows.")

            # 🔥 HIGH FEATURE CHECK
            max_features = 1000
            if X.shape[1] > max_features:
                st.warning(f"High dimensional dataset detected ({X.shape[1]} features).")
                st.info("Models may be slow or overfit. Consider reducing features.")

            train_clicked = st.button("🚀 Train Model", disabled=not dataset_valid)
            if train_clicked:

                if X.isnull().sum().sum() > 0:
                    st.error("❌ Dataset contains missing values.")
                    st.stop()

                if not all(X.dtypes.apply(lambda x: x.kind in "iufcb")):
                    st.error("❌ Dataset contains non-numeric columns.")
                    st.stop()

                try:
                    # 🔴 Safe Train-Test Split
                    if problem_type == "Classification":
                        class_counts = y.value_counts()

                        if class_counts.min() < 2:
                            st.error("❌ Each class must have at least 2 samples for train-test split.")
                            st.stop()

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )
                    # 🔴 Post-Split Class Check
                    if problem_type == "Classification" and y_train.nunique() < 2:
                        st.error("❌ Training data contains only one class after split.")
                        st.stop()

                    if model_name in ["Logistic Regression", "KNN", "KNN Regressor", "Linear Regression", "SVM", "SVR"]:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Model creation
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(random_state=42,**params)

                    elif model_name == "KNN":
                        if params["n_neighbors"] > len(X_train):
                            params["n_neighbors"] = len(X_train)
                            st.warning(f"K automatically adjusted to {len(X_train)}")
                        model = KNeighborsClassifier(**params)

                    elif model_name == "KNN Regressor":
                        if params["n_neighbors"] > len(X_train):
                            params["n_neighbors"] = len(X_train)
                            st.warning(f"K automatically adjusted to {len(X_train)}")
                        model = KNeighborsRegressor(**params)

                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(
                            random_state=42,
                            n_jobs=-1,
                            **params
                        )

                    elif model_name == "Linear Regression":
                        model = LinearRegression()

                    elif model_name == "Decision Tree Regressor":
                        model = DecisionTreeRegressor(**params)

                    elif model_name == "Decision Tree":
                        model = DecisionTreeClassifier(**params)

                    elif model_name == "Random Forest Regressor":
                        model = RandomForestRegressor(
                            random_state=42,
                            n_jobs=-1,
                            **params
                        )

                    elif model_name == "SVM":
                        model = SVC(**params)

                    elif model_name == "SVR":
                        model = SVR(**params)   

                    # Training
                    with st.spinner("⏳ Please wait... Model is training..."):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    st.success("✅ Model Trained Successfully!")
                    st.subheader("📊 Model Results")

                    # ================= CLASSIFICATION =================

                    if problem_type == "Classification":

                        acc = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", round(acc, 4))

                        cm = confusion_matrix(y_test, y_pred)

                        fig, ax = plt.subplots()
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="viridis",
                            xticklabels=model.classes_,
                            yticklabels=model.classes_,
                            ax=ax
                        )
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)

                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.2f}"))

                    # ================= REGRESSION =================

                    else:

                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mse)

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("R2 Score", round(r2, 4))
                        col2.metric("MSE", round(mse, 4))
                        col3.metric("RMSE", round(rmse, 4))
                        col4.metric("MAE", round(mae, 4))

                except Exception as e:
                    st.error("❌ Model Training Failed!")
                    with st.expander("🔍 See Technical Error"):
                        st.code(str(e))
                
            if not dataset_valid:
                st.info("Fix the above issues to enable model training.")
else:
    st.info("Upload a CSV file to begin.")
