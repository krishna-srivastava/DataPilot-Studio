# 🛸 DataPilot Studio  
> **Upload CSV → Explore Data → Generate Insights**

A powerful, dark-themed **Exploratory Data Analysis (EDA)** and **Data Cleaning** web app built with Python & Streamlit. Designed for data analysts, students, and ML practitioners who want fast, visual insights from their CSV datasets — no code required.

---

## ✨ Features

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

> Max upload size: **50MB CSV**

---

## App Preview:
![DataPilot Studio Screenshot](app_screenshot.png)

---

## 🙋‍♂️ Author 
**Krishna Srivastava**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/krishna-srivastava-b402a1323)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/krishna-srivastava)
