import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mine Workers Monitoring System",
    page_icon="⛏️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    font-size: 16px;
    margin-bottom: 25px;
}

.normal-box {
    padding: 18px;
    border-radius: 12px;
    background: #e8f5e9;
    border-left: 6px solid #2e7d32;
}

.warning-box {
    padding: 18px;
    border-radius: 12px;
    background: #fff8e1;
    border-left: 6px solid #f57c00;
}

.critical-box {
    padding: 18px;
    border-radius: 12px;
    background: #ffebee;
    border-left: 6px solid #c62828;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    background: #f5f5f5;
    text-align: center;
}

.reason {
    padding: 8px;
    margin: 4px 0;
    border-radius: 6px;
    background: #f7f7f7;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">⛏️ Mine Workers Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Health, environmental gas and machine-learning monitoring dashboard</div>',
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, names):
    """Find a column using exact or partial matching."""
    columns = {str(c).strip().lower(): c for c in df.columns}

    # Exact match
    for name in names:
        key = name.strip().lower()
        if key in columns:
            return columns[key]

    # Partial match
    for column in df.columns:
        column_text = str(column).strip().lower()

        for name in names:
            name_text = name.strip().lower()

            if name_text in column_text:
                return column

    return None


def clean_numeric(series):
    """Convert a column to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def display_value(value):
    """Display a clean value."""
    if pd.isna(value):
        return "N/A"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

health_parameters = {
    "Heart Rate": [
        "Heart_Rate",
        "Heart Rate",
        "HeartRate",
        "HR"
    ],
    "SpO₂": [
        "SpO2",
        "SpO₂",
        "SPO2",
        "Oxygen Saturation"
    ],
    "Temperature": [
        "Temperature",
        "Temp",
        "Body Temperature"
    ],
    "Blood Pressure": [
        "Blood Pressure",
        "Blood_Pressure",
        "BP"
    ],
    "Pulse": [
        "Pulse",
        "Pulse_Rate",
        "Pulse Rate"
    ],
    "Respiratory Rate": [
        "Respiratory_Rate",
        "Respiratory Rate",
        "Respiration Rate",
        "RR"
    ]
}

gas_parameters = {
    "CH₄ Methane": [
        "CH4",
        "CH₄",
        "Methane"
    ],
    "CO Carbon Monoxide": [
        "CO",
        "Carbon Monoxide"
    ],
    "CO₂ Carbon Dioxide": [
        "CO2",
        "CO₂",
        "Carbon Dioxide"
    ],
    "H₂S Hydrogen Sulfide": [
        "H2S",
        "H₂S",
        "Hydrogen Sulfide"
    ],
    "O₂ Oxygen": [
        "O2",
        "O₂",
        "Oxygen"
    ]
}

status_columns = [
    "Status",
    "Condition",
    "Risk",
    "Risk_Level",
    "Risk Level",
    "Class",
    "Label",
    "Target",
    "Status_Label"
]

worker_columns = [
    "Worker_ID",
    "Worker ID",
    "worker_id",
    "Worker",
    "Worker Name",
    "Name",
    "ID"
]


# ============================================================
# DATA UPLOAD
# ============================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Worker Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info(
        "👆 Upload a CSV or Excel worker dataset from the sidebar to start monitoring."
    )

    st.markdown("### 📋 Expected Data")

    st.write(
        "The application can detect common column names for worker identity, "
        "health parameters, gas levels and condition labels."
    )

    st.stop()


# ============================================================
# READ DATASET
# ============================================================

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

except Exception as error:
    st.error(f"Unable to read the dataset: {error}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# Remove completely empty columns
df = df.dropna(axis=1, how="all").copy()

# ============================================================
# IDENTIFY COLUMNS
# ============================================================

worker_col = find_column(df, worker_columns)
status_col = find_column(df, status_columns)

health_cols = {}
for label, possible_names in health_parameters.items():
    health_cols[label] = find_column(df, possible_names)

gas_cols = {}
for label, possible_names in gas_parameters.items():
    gas_cols[label] = find_column(df, possible_names)


# ============================================================
# WORKER ID
# ============================================================

if worker_col is None:
    df["Worker_ID_Auto"] = [
        f"Worker-{i + 1:03d}" for i in range(len(df))
    ]
    worker_col = "Worker_ID_Auto"


df[worker_col] = df[worker_col].astype(str)


# ============================================================
# PREPARE NUMERIC FEATURES
# ============================================================

feature_columns = []

for column in list(health_cols.values()) + list(gas_cols.values()):
    if column is not None:
        if column not in feature_columns:
            feature_columns.append(column)

for column in feature_columns:
    df[column] = clean_numeric(df[column])


# ============================================================
# CONDITION MAPPING
# ============================================================

def map_condition(value):
    text = str(value).strip().upper()

    if text in ["NORMAL", "LOW", "SAFE", "0"]:
        return "🟢 NORMAL"

    if text in ["WARNING", "WARN", "MEDIUM", "MODERATE", "1"]:
        return "🟠 WARNING"

    if text in ["CRITICAL", "CRIT", "HIGH", "DANGER", "2"]:
        return "🔴 CRITICAL"

    return None


# ============================================================
# MACHINE LEARNING
# ============================================================

model_used = "None"
ml_reasons = [[] for _ in range(len(df))]

if status_col is not None:
    mapped_labels = df[status_col].apply(map_condition)

    valid_labels = mapped_labels.notna()

    if valid_labels.sum() >= 3 and valid_labels.nunique() >= 2:
        model_features = feature_columns.copy()

        if len(model_features) > 0:
            X = df[model_features].copy()

            # Fill missing values using column medians
            for column in model_features:
                median = X[column].median()

                if pd.isna(median):
                    median = 0

                X[column] = X[column].fillna(median)

            y = mapped_labels.copy()

            # Train only on rows with usable labels
            X_train = X.loc[valid_labels]
            y_train = y.loc[valid_labels]

            try:
                model = RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced"
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X)

                df["Overall Condition"] = predictions

                model_used = "Random Forest"

            except Exception:
                df["Overall Condition"] = mapped_labels

        else:
            df["Overall Condition"] = mapped_labels

    else:
        df["Overall Condition"] = None

else:
    df["Overall Condition"] = None


# ============================================================
# ISOLATION FOREST FALLBACK
# ============================================================

if df["Overall Condition"].isna().all() and len(feature_columns) > 0:

    X = df[feature_columns].copy()

    for column in feature_columns:
        median = X[column].median()

        if pd.isna(median):
            median = 0

        X[column] = X[column].fillna(median)

    try:
        isolation_model = IsolationForest(
            n_estimators=150,
            contamination="auto",
            random_state=42
        )

        anomaly_prediction = isolation_model.fit_predict(X)
        anomaly_score = -isolation_model.decision_function(X)

        df["ML Anomaly Score"] = anomaly_score

        # Rank workers by anomaly score
        ranks = pd.Series(anomaly_score).rank(
            pct=True,
            method="average"
        )

        conditions = []

        for rank in ranks:
            if rank >= 0.90:
                conditions.append("🔴 CRITICAL")
            elif rank >= 0.70:
                conditions.append("🟠 WARNING")
            else:
                conditions.append("🟢 NORMAL")

        df["Overall Condition"] = conditions

        model_used = "Isolation Forest"

        # Identify strongest numeric deviations for explanations
        for i in range(len(df)):

            row_reasons = []

            for column in feature_columns:

                values = df[column]

                median = values.median()
                mad = (values - median).abs().median()

                if pd.isna(median):
                    continue

                if pd.isna(mad) or mad == 0:
                    continue

                value = df.iloc[i][column]

                if pd.isna(value):
                    continue

                robust_score = abs(value - median) / (mad * 1.4826)

                if robust_score >= 3:
                    row_reasons.append(
                        f"{column}: {display_value(value)} differs strongly from the dataset pattern"
                    )

            ml_reasons[i] = row_reasons[:5]

    except Exception:
        df["Overall Condition"] = "🟢 NORMAL"
        model_used = "Fallback"


# ============================================================
# FINAL FALLBACK
# ============================================================

df["Overall Condition"] = df["Overall Condition"].fillna(
    "🟢 NORMAL"
)


# ============================================================
# CONDITION REASONS
# ============================================================

def get_reasons(row):

    reasons = []

    condition = row["Overall Condition"]

    # Dataset label explanation
    if status_col is not None:
        original = row[status_col]

        if not pd.isna(original):
            reasons.append(
                f"Dataset condition label: {original}"
            )

    # ML explanations
    row_index = row.name

    if row_index < len(ml_reasons):
        reasons.extend(ml_reasons[row_index])

    # Missing monitoring values
    missing_health = []

    for label, column in health_cols.items():
        if column is not None:
            if pd.isna(row[column]):
                missing_health.append(label)

    if missing_health:
        reasons.append(
            "Missing health data: " + ", ".join(missing_health)
        )

    missing_gases = []

    for label, column in gas_cols.items():
        if column is not None:
            if pd.isna(row[column]):
                missing_gases.append(label)

    if missing_gases:
        reasons.append(
            "Missing gas data: " + ", ".join(missing_gases)
        )

    if not reasons:
        if condition == "🟢 NORMAL":
            reasons.append(
                "No significant anomaly was identified by the configured monitoring model."
            )

        elif condition == "🟠 WARNING":
            reasons.append(
                "The monitoring model identified a moderate deviation from the dataset pattern."
            )

        elif condition == "🔴 CRITICAL":
            reasons.append(
                "The monitoring model identified a strong deviation from the dataset pattern."
            )

    return reasons


df["Condition Reasons"] = df.apply(get_reasons, axis=1)


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

st.markdown("## 📊 Overall Monitoring")

total_workers = len(df)

normal_count = int(
    (df["Overall Condition"] == "🟢 NORMAL").sum()
)

warning_count = int(
    (df["Overall Condition"] == "🟠 WARNING").sum()
)

critical_count = int(
    (df["Overall Condition"] == "🔴 CRITICAL").sum()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👷 Total Workers", total_workers)

with col2:
    st.metric("🟢 Normal", normal_count)

with col3:
    st.metric("🟠 Warning", warning_count)

with col4:
    st.metric("🔴 Critical", critical_count)


st.caption(f"ML method used: **{model_used}**")


# ============================================================
# ALL WORKERS
# ============================================================

st.markdown("## 👷 All Workers Monitoring")

search_text = st.text_input(
    "🔎 Search Worker",
    placeholder="Enter Worker ID or Worker Name..."
)

filtered_df = df.copy()

if search_text.strip():

    search_text = search_text.strip().lower()

    mask = filtered_df.apply(
        lambda row: row.astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
        .any(),
        axis=1
    )

    filtered_df = filtered_df[mask]


display_columns = [worker_col, "Overall Condition"]

# Add available health columns
for label, column in health_cols.items():
    if column is not None:
        display_columns.append(column)

# Add available gas columns
for label, column in gas_cols.items():
    if column is not None:
        display_columns.append(column)

display_columns = list(dict.fromkeys(display_columns))

table_df = filtered_df[display_columns].copy()

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONDITION CHART
# ============================================================

st.markdown("## 📈 Condition Summary")

chart_data = pd.DataFrame({
    "Condition": [
        "🟢 NORMAL",
        "🟠 WARNING",
        "🔴 CRITICAL"
    ],
    "Workers": [
        normal_count,
        warning_count,
        critical_count
    ]
})

fig = px.bar(
    chart_data,
    x="Condition",
    y="Workers",
    text="Workers",
    title="Worker Condition Distribution"
)

fig.update_layout(
    xaxis_title="Condition",
    yaxis_title="Number of Workers"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL WORKER
# ============================================================

st.markdown("## 👤 Individual Worker Monitoring")

if len(filtered_df) == 0:

    st.warning("No workers match your search.")

else:

    worker_options = filtered_df[worker_col].tolist()

    selected_worker = st.selectbox(
        "Select Worker",
        worker_options
    )

    worker_row = filtered_df[
        filtered_df[worker_col] == selected_worker
    ].iloc[0]

    condition = worker_row["Overall Condition"]

    st.markdown(f"### Worker: **{selected_worker}**")

    # ========================================================
    # CONDITION MESSAGE
    # ========================================================

    if condition == "🟢 NORMAL":

        st.markdown(
            """
            <div class="normal-box">
            <h3>🟢 NORMAL</h3>
            Worker is currently classified as NORMAL by the configured
            monitoring model.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif condition == "🟠 WARNING":

        st.markdown(
            """
            <div class="warning-box">
            <h3>🟠 WARNING</h3>
            The monitoring model identified a moderate deviation.
            Review the listed monitoring reasons.
            </div>
            """,
            unsafe_allow_html=True
        )

    elif condition == "🔴 CRITICAL":

        st.markdown(
            """
            <div class="critical-box">
            <h3>🔴 CRITICAL</h3>
            The monitoring model identified a strong deviation.
            Review the listed monitoring reasons and follow applicable
            workplace safety procedures.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # REASONS
    # ========================================================

    st.markdown("### ⚠️ Conditions / Reasons")

    reasons = worker_row["Condition Reasons"]

    for reason in reasons:
        st.markdown(
            f'<div class="reason">• {reason}</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # HEALTH PARAMETERS
    # ========================================================

    st.markdown("### ❤️ Health Parameters")

    health_items = []

    for label, column in health_cols.items():

        if column is not None:

            value = worker_row[column]

            health_items.append(
                (label, display_value(value))
            )

    if health_items:

        columns = st.columns(min(3, len(health_items)))

        for index, (label, value) in enumerate(health_items):

            with columns[index % len(columns)]:
                st.metric(
                    label,
                    value
                )

    else:

        st.info("No recognized health parameter columns were found.")


    # ========================================================
    # GAS LEVELS
    # ========================================================

    st.markdown("### 🌫️ Gas Levels")

    gas_items = []

    for label, column in gas_cols.items():

        if column is not None:

            value = worker_row[column]

            gas_items.append(
                (label, display_value(value))
            )

    if gas_items:

        columns = st.columns(min(3, len(gas_items)))

        for index, (label, value) in enumerate(gas_items):

            with columns[index % len(columns)]:
                st.metric(
                    label,
                    value
                )

    else:

        st.info("No recognized gas columns were found.")


    # ========================================================
    # WORKER DATA
    # ========================================================

    st.markdown("### 📋 Complete Worker Record")

    complete_record = worker_row.drop(
        labels=["Condition Reasons"],
        errors="ignore"
    )

    record_df = pd.DataFrame(
        {
            "Parameter": complete_record.index.astype(str),
            "Value": complete_record.values
        }
    )

    st.dataframe(
        record_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⛏️ Mine Workers Monitoring System • Python • Streamlit • Scikit-learn"
)

st.caption(
    "Monitoring indicators are based on the uploaded dataset and configured "
    "machine-learning analysis. They are not medical diagnoses."
)