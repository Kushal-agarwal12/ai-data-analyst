import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

def load_dataset(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    if uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if uploaded_file.name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file type")

def profile_dataset(df: pd.DataFrame) -> dict:
    missing = df.isna().mean().sort_values(ascending=False)
    duplicates = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    data_health = 100
    data_health -= min(40, int(missing.mean() * 100))
    data_health -= min(20, int(duplicates / max(1, len(df)) * 100))
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing": missing,
        "duplicates": duplicates,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "health_score": max(0, data_health),
    }

def classify_columns(missing: pd.Series, threshold: float = 0.2) -> tuple[list[str], list[str]]:
    bad = missing[missing > threshold].index.tolist()
    good = missing[missing <= threshold].index.tolist()
    return good, bad

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool,
    drop_missing_rows: bool,
    drop_missing_cols: bool,
    missing_row_threshold: float,
    missing_col_threshold: float,
    impute_numeric: bool,
    impute_categorical: bool,
) -> tuple[pd.DataFrame, list[str]]:
    changes = []
    cleaned = df.copy()

    if drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)
        changes.append(f"Removed {removed} duplicate rows.")

    if drop_missing_cols:
        missing_ratio = cleaned.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > missing_col_threshold].index.tolist()
        if cols_to_drop:
            cleaned = cleaned.drop(columns=cols_to_drop)
            changes.append(f"Dropped columns with > {missing_col_threshold:.0%} missing: {', '.join(cols_to_drop)}.")

    if drop_missing_rows:
        before = len(cleaned)
        cleaned = cleaned[cleaned.isna().mean(axis=1) <= missing_row_threshold]
        removed = before - len(cleaned)
        changes.append(f"Removed {removed} rows with > {missing_row_threshold:.0%} missing values.")

    if impute_numeric:
        numeric_cols = cleaned.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if cleaned[col].isna().any():
                median = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median)
        changes.append("Imputed missing numeric values with median.")

    if impute_categorical:
        categorical_cols = cleaned.select_dtypes(exclude=np.number).columns
        for col in categorical_cols:
            if cleaned[col].isna().any():
                mode = cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown"
                cleaned[col] = cleaned[col].fillna(mode)
        changes.append("Imputed missing categorical values with mode.")

    return cleaned, changes

def naive_sentiment_score(text: str) -> float:
    positive_words = {"good", "great", "excellent", "positive", "happy", "love", "success"}
    negative_words = {"bad", "poor", "negative", "sad", "hate", "fail", "issue"}
    tokens = {token.strip(".,!?;:").lower() for token in str(text).split()}
    score = sum(1 for t in tokens if t in positive_words) - sum(1 for t in tokens if t in negative_words)
    return score

def generate_report(summary: dict, cleaning_log: list[str], analysis_notes: list[str]) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report = [
        "# Executive Report",
        f"Generated: {timestamp}",
        "",
        "## Dataset Overview",
        f"- Rows: {summary['rows']}",
        f"- Columns: {summary['columns']}"
        f"- Duplicate rows: {summary['duplicates']}"
        f"- Data health score: {summary['health_score']}",
        "",
        "## Data Quality & Cleaning Actions",
    ]
    if cleaning_log:
        report.extend([f"- {entry}" for entry in cleaning_log])
    else:
        report.append("- No cleaning actions applied.")
    report.extend(["", "## Analysis Findings"])
    if analysis_notes:
        report.extend([f"- {note}" for note in analysis_notes])
    else:
        report.append("- No analysis findings recorded.")
    report.extend([
        "",
        "## Recommendations",
        "- Review KPI trends and focus on segments with performance gaps.",
        "- Validate anomalies with domain experts before taking action.",
    ])
    return "\n".join(report)

def render_dashboard(df: pd.DataFrame, profile: dict) -> None:
    st.subheader("Decision Dashboard")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Rows", profile["rows"])
    kpi2.metric("Columns", profile["columns"])
    kpi3.metric("Data Health", profile["health_score"])

    filter_columns = df.select_dtypes(exclude=np.number).columns.tolist()
    if filter_columns:
        column = st.selectbox("Filter by", filter_columns)
        values = st.multiselect("Values", df[column].dropna().unique().tolist())
        if values:
            df = df[df[column].isin(values)]

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 1:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="KPI Relationship")
        st.plotly_chart(fig, use_container_width=True)


st.title("AI Data Analyst Agent")

uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "json", "parquet"])

if uploaded_file:
    data = load_dataset(uploaded_file)
    st.success(f"Loaded {uploaded_file.name}")

    overview_tab, cleaning_tab, analysis_tab, dashboard_tab, report_tab = st.tabs(
        ["Overview", "Cleaning", "Analysis", "Dashboard", "Report"]
    )

    with overview_tab:
        st.subheader("Dataset Overview")
        profile = profile_dataset(data)
        st.write(f"Rows: {profile['rows']} | Columns: {profile['columns']} | Data Health Score: {profile['health_score']}")

        good_cols, bad_cols = classify_columns(profile["missing"], threshold=0.2)
        left, right = st.columns(2)
        with left:
            st.markdown("**Good Sections (low missingness)**")
            st.write(good_cols)
        with right:
            st.markdown("**Bad Sections (needs cleaning)**")
            st.write(bad_cols)

        st.dataframe(data.head())
        st.markdown("### Missing Values by Column")
        st.bar_chart(profile["missing"])

    with cleaning_tab:
        st.subheader("Cleaning Controls")
        drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
        drop_missing_rows = st.checkbox("Drop rows with high missingness", value=False)
        drop_missing_cols = st.checkbox("Drop columns with high missingness", value=False)
        missing_row_threshold = st.slider("Row missingness threshold", 0.0, 1.0, 0.5, 0.05)
        missing_col_threshold = st.slider("Column missingness threshold", 0.0, 1.0, 0.5, 0.05)
        impute_numeric = st.checkbox("Impute numeric values", value=True)
        impute_categorical = st.checkbox("Impute categorical values", value=True)

        cleaned, change_log = clean_dataset(
            data,
            drop_duplicates,
            drop_missing_rows,
            drop_missing_cols,
            missing_row_threshold,
            missing_col_threshold,
            impute_numeric,
            impute_categorical,
        )

        st.markdown("### Cleaning Summary")
        for entry in change_log:
            st.write(f"- {entry}")

        st.markdown("### Before vs After")
        col1, col2 = st.columns(2)
        col1.metric("Original rows", len(data))
        col2.metric("Cleaned rows", len(cleaned))

        st.dataframe(cleaned.head())
        buffer = io.BytesIO()
        cleaned.to_csv(buffer, index=False)
        st.download_button("Download cleaned dataset", data=buffer.getvalue(), file_name="cleaned_dataset.csv")

    with analysis_tab:
        st.subheader("Exploratory Data Analysis")
        analysis_notes: list[str] = []
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            fig = px.histogram(data, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
            analysis_notes.append(f"Reviewed distribution for {col}.")

        if len(numeric_cols) > 1:
            corr = data[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            analysis_notes.append("Checked correlations among numeric features.")

        text_cols = data.select_dtypes(include="object").columns
        if len(text_cols) > 0:
            st.markdown("### Naive Sentiment Signals")
            text_col = st.selectbox("Text column", text_cols)
            data["sentiment_score"] = data[text_col].fillna("").apply(naive_sentiment_score)
            fig = px.histogram(data, x="sentiment_score", title="Sentiment Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
            analysis_notes.append(f"Generated sentiment distribution for {text_col}.")

        datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            st.markdown("### Time Series Snapshot")
            date_col = datetime_cols[0]
            metric = numeric_cols[0]
            time_df = data[[date_col, metric]].dropna().sort_values(date_col)
            fig = px.line(time_df, x=date_col, y=metric, title=f"{metric} over time")
            st.plotly_chart(fig, use_container_width=True)
            analysis_notes.append(f"Plotted {metric} over time based on {date_col}.")

    with dashboard_tab:
        render_dashboard(data, profile_dataset(data))

    with report_tab:
        st.subheader("Executive Report")
        profile_summary = profile_dataset(data)
        report_text = generate_report(profile_summary, change_log, analysis_notes)
        st.text_area("Report Preview", report_text, height=400)
        st.download_button("Download report", data=report_text, file_name="executive_report.md")
else:
    st.info("Upload a dataset to begin.")
