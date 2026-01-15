# AI Data Analyst Agent

## Goal
Build an AI-first data analyst agent that guides any company from raw data upload to a polished, executive-grade report. The agent should be **highly trained** to understand datasets, assess data quality, recommend/perform the right analyses, and generate decision-ready dashboards and reports.

## End-to-End User Flow
1. **Upload** a dataset.
2. **Understand & profile** the dataset (automated overview + data quality health).
3. **Clean** the dataset (handle missing values, duplicates, formats).
4. **Analyze** (EDA + suggested advanced analyses with visuals and explanations).
5. **Dashboard** (interactive KPI-driven decision dashboard).
6. **Report** (executive summary, methods, findings, recommendations).
7. **Export** the cleaned dataset and analysis assets.

---

## 1) Dataset Upload & Understanding
The agent must accept common data formats:
- CSV, XLSX, JSON, Parquet

### Dataset Overview (presented like a dashboard)
The first screen must be a **highly presentable data health overview** with:
- **Composition**: rows, columns, data types, memory usage
- **Summary stats**: numeric distribution, categorical distribution
- **Column profiling**: unique counts, cardinality, missingness
- **Data quality & health**:
  - Missing values
  - Duplicates
  - Outliers (where applicable)
  - Inconsistent formats (dates, currency, casing, etc.)
- **Good sections / bad sections**: highlight clean vs. problematic columns/rows
- **Data health score**: a single roll-up score with contributing factors

> The overview must clearly explain what needs cleaning, why it matters, and the expected impact.

---

## 2) Data Cleaning Engine
The agent must automatically clean the dataset after the overview is approved:
- **Handle missing values** (smart imputation based on column type)
- **Drop or merge duplicates**
- **Fix formatting** (dates, currency, categorical standardization)
- **Correct invalid values** (out-of-range, inconsistent categories)
- **Log every change** in a reproducible audit trail

### User Controls
- Allow the user to **accept, customize, or skip** each cleaning step.
- Provide **before/after comparisons** for every change.

### Output
- Export the **cleaned dataset**
- Provide a **cleaning summary report** (before vs. after quality metrics)

---

## 3) Analysis Section
The agent must perform **EDA for every dataset**, plus recommend and execute the most suitable advanced analyses based on the dataset schema.

### Required Analyses
- **EDA (always)**: distributions, correlations, pairwise relationships, anomalies
- **Descriptive analysis**: summary trends and key metrics

### Conditional Analyses (auto-suggested)
- **Time series analysis**: if date/time columns exist
- **Sentiment analysis**: if text columns exist
- **Cohort/retention analysis**: if user/customer + time columns exist
- **Forecasting**: if historical time-based metrics exist
- **Segmentation**: if behavioral or categorical clusters exist
- **Anomaly detection**: if KPI drift or outliers are likely

### Recommendations Engine
- Explain **why** specific analyses are recommended based on the dataset headers/columns.
- Allow users to **run any analysis** on demand, even if not recommended.

### Presentation Requirements
- Each analysis must include:
  - Clear visuals
  - Brief, plain-language interpretation
  - Actionable insights
  - Rationale for why this analysis was chosen

---

## 4) Dashboard Section (Decision-Making Focus)
A **Power BI–style interactive dashboard** that is built from:
- The uploaded dataset
- Cleaned data outputs
- Analysis results

### Must Include
- KPI cards
- Filters (dynamic + multi-select)
- Interactive charts
- Drill-down views
- Decision-ready summary panels

---

## 5) Reporting System
The agent must generate a structured, professional executive report:

### Required Sections
1. **Executive Summary**
2. **Dataset Overview**
3. **Data Quality & Cleaning Actions**
4. **Analysis Findings**
5. **Dashboards & KPIs**
6. **Recommendations**
7. **Appendix** (methods, assumptions, data dictionary)

The report must be formatted and professional — suitable for leadership.

---

## Training Requirements (Critical)
The agent must be **heavily trained** to:
- Understand schema and data types automatically
- Detect data quality issues reliably
- Choose the right analytics method for the dataset
- Explain results in business language
- Produce polished dashboards and executive reports

### Accuracy Requirements
- Validate results against known benchmarks on curated datasets.
- Provide confidence levels for analysis outputs.

---

## Export & Collaboration
- Export cleaned data (CSV/Parquet)
- Export analysis visuals
- Export full report (PDF/DOCX)
- Provide notebooks or reproducible scripts

---

## Success Criteria
The system is successful if:
- A non-technical user can upload data and receive a complete executive report
- Insights are accurate, visual, and business-ready
- Data quality issues are correctly flagged and corrected
- The dashboard supports real decision-making

---

## Next Steps
- Define the tech stack (e.g., Python + Pandas + Plotly + Streamlit)
- Specify UI/UX design for the overview and dashboard
- Train the agent with business datasets across industries
- Implement prompt and analysis evaluation benchmarks
