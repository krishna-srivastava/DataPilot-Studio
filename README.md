# 🛸 DataPilot Studio  
> **Upload CSV → Explore & Clean → Train Models**

A powerful, dark-themed **EDA + Data Cleaning + Model Training** web app built with Python & Streamlit. Designed for data analysts, students, and ML practitioners who want to go from raw CSV to a trained, downloadable model — no code required.

The app is organized into two labs: **🔭 EDA Lab** for exploring and cleaning your data, and **🤖 Model Lab** for encoding, feature importance, and training ML models.

---

## ✨ Features

## 🔭 EDA Lab

### 📊 Overview Tab
- Dataset preview (first 10 rows)
- Column names & data types
- Statistical summary — Numerical, Categorical, and Boolean columns separately
- Missing values table with percentage
- **Data Health Score** — Scores dataset quality out of 100 based on missing values, duplicates, and outliers
- **Outlier Detection** using IQR method — Lower/Upper bounds, count & percentage per column

### 🔍 Column Analyzer Tab 
- Per-column deep dive — Data type, missing %, unique values
- **Numeric columns:** Mean, Median, Std Dev, Min, Max, Skewness, Kurtosis
- **Skewness & Kurtosis interpretation** — Automatically labels distribution shape (Symmetric, Right Skewed, Leptokurtic, etc.), with safe handling for constant or all-null columns
- Distribution histogram + KDE curve
- Box plot with outlier markers
- Outlier bounds (IQR) per column
- **Categorical columns:** Most frequent value, Value distribution table with %, Bar chart, Pie chart

### 🔗 Correlation Tab
- **Target-driven selection** — Pick a target column, then choose which numeric/boolean columns to compare it against (or Select All, capped at 20 columns for readability and performance)
- **Correlation-with-target table** — Sorted by strength, color-coded (blue = positive, red = negative)
- **Full correlation matrix table** — Every selected pair, color-coded the same way
- **Correlation heatmap** — Annotated for ≤15 columns, with the target row/column highlighted
- Correlation guide banner (+1 / 0 / -1 explanation, strength thresholds)

### 🧹 Data Cleaning Tab
- Missing values table (only affected columns shown)
- **Smart fill** — Mean/Median/Mode for numeric, Mode only for categorical
- **Custom value fill** — Enter any value manually
- **Rename columns**
- **Delete columns** (disabled when only one column remains)
- **Tab-scoped Undo** — Step-by-step undo for fill/rename/delete actions on this tab only
- Reset to original dataset
- Download cleaned CSV

### 🗂️ Duplicate Rows Tab
- Check duplicates — All columns or specific column
- Stats: Total rows, Duplicate count, Duplicate %
- Preview duplicate rows
- Delete all duplicates
- **Tab-scoped Undo** & Reset support (independent from the Data Cleaning tab's undo history)
- Download cleaned CSV

### 📈 Visualization Tab
- 7 chart types: Histogram, Box Plot, Bar Chart, Scatter Plot, Line Chart, Violin Plot, Pie Chart
- **Auto column filter** — Only valid columns shown per chart type (no more wrong column errors)
- Color-by (hue) support for Scatter Plot
- Auto sampling for large datasets (>10,000 rows)
- Graceful handling of empty/constant columns instead of crashing
- Download any chart as PNG

## 🤖 Model Lab

### 🔤 Encoding Tab
- Auto-detects all categorical columns with unique-value and missing-value counts
- **Label Encoding**, **One-Hot Encoding** (capped at 25 unique values), and **Manual Ordinal Encoding** (define your own low-to-high order)
- Blocks encoding on columns with missing values until they're cleaned
- **Tab-scoped Undo & Reset** — independent of the EDA Lab's cleaning history
- Download encoded CSV

### 🌲 Feature Importance Tab
- Auto-detects Classification vs Regression based on target uniqueness
- Runs a Random Forest to rank every feature by importance
- Top-3 feature podium, full ranked table with cumulative %, and a bar chart
- Auto-samples large datasets (>100,000 rows) for speed

### 🎯 Training Tab
- Auto-detects Classification vs Regression from the selected target
- **Models:** Linear/Logistic Regression, KNN, Decision Tree, Random Forest, **XGBoost**, **LightGBM** (XGBoost/LightGBM auto-hide if not installed)
- Adjustable train/test split with row-count preview
- **Hyperparameter tuning** per model (max depth, n_estimators, learning rate, kernel, etc.) with sensible defaults when tuning is off
- Auto-scaling for distance-based models, auto-PCA for high-dimensional data (>100 features, 90% variance retained)
- **Stratified train/test split** for classification when every class has enough samples
- Regression results: R², MAE, MSE, RMSE + Actual vs Predicted plot
- Classification results: Accuracy, full classification report, confusion matrix
- 5-fold cross-validation with per-fold chart
- Download the trained model bundle (model + scaler + PCA) as a `.pkl` file

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | Web app framework |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Chart rendering |
| Seaborn | Statistical visualizations |
| PyArrow | Fast CSV loading |
| scikit-learn | Encoding, feature importance, model training, metrics |
| XGBoost *(optional)* | Gradient boosting model |
| LightGBM *(optional)* | Gradient boosting model |

> XGBoost and LightGBM are optional — if not installed, they're simply hidden from the model list in the Training tab.

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/krishna-srivastava/DataPilot-Studio.git
cd datapilot-studio
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

> Max upload size: **25MB CSV**

---

## App Preview:
![DataPilot Studio Screenshot](app_screenshot.png)

---

## 🙋‍♂️ Author 
**Krishna Srivastava**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/krishna-srivastava-b402a1323)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/krishna-srivastava)
