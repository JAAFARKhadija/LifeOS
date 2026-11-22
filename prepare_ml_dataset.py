
import pandas as pd
from pathlib import Path

# 1. Load original dataset
file_path = Path("data/synthetic/study_tasks.csv")

df = pd.read_csv(
    file_path,
    parse_dates=["created_date", "deadline", "completion_date"]
)

print("Original dataset:", df.shape)

# 2. Exclude pending tasks
df = df[df["outcome"] != "Pending"].copy()

# 3. Create binary target
df["completed_on_time"] = (
    df["completion_date"].notna()
    & (df["completion_date"] <= df["deadline"])
).astype(int)


# 4. Feature engineering

# How many days are available before the deadline?
df["days_until_deadline"] = (
    df["deadline"] - df["created_date"]
).dt.days

# Day of the week: Monday = 0, Sunday = 6
df["created_weekday"] = df["created_date"].dt.dayofweek

# Month: January = 1, December = 12
df["created_month"] = df["created_date"].dt.month

# 5. Select columns for ML
ml_df = df[
    [
        "task_id",
        "created_date",
        "subject",
        "task_type",
        "priority",
        "estimated_duration",
        "deadline",
        "days_until_deadline",
        "created_weekday",
        "created_month",
        "completed_on_time",
        "is_synthetic"
    ]
].copy()


# 5. Save prepared dataset
output_path = Path("data/synthetic/study_tasks_ml.csv")
ml_df.to_csv(output_path, index=False)

# 6. Display results
print("\nML DATASET CREATED")
print("Shape:", ml_df.shape)

print("\nTarget distribution:")
print(ml_df["completed_on_time"].value_counts())

print("\nTarget percentages:")
print(
    (ml_df["completed_on_time"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nSaved to:", output_path)


# ==========================================
# 7. FEATURE VALIDATION
# ==========================================

print("\nFEATURE VALIDATION")

print("\nNumerical feature statistics:")
print(
    ml_df[
        [
            "estimated_duration",
            "days_until_deadline",
            "created_weekday",
            "created_month"
        ]
    ].describe()
)

print("\nWeekday distribution:")
print(ml_df["created_weekday"].value_counts().sort_index())

print("\nMonth distribution:")
print(ml_df["created_month"].value_counts().sort_index())

print("\nSubject distribution:")
print(ml_df["subject"].value_counts())

# Basic validation
assert ml_df["days_until_deadline"].between(1, 10).all()
assert ml_df["created_weekday"].between(0, 6).all()
assert ml_df["created_month"].between(1, 12).all()
assert ml_df["estimated_duration"].between(30, 420).all()

print("\nAll feature validation checks passed!")


# ==========================================
# 8. DEFINE FEATURES AND TARGET
# ==========================================

# Features: information available when a task is created
feature_columns = [
    "subject",
    "task_type",
    "priority",
    "estimated_duration",
    "days_until_deadline",
    "created_weekday"
]

X = ml_df[feature_columns].copy()

# Target: what we want the model to predict
y = ml_df["completed_on_time"].copy()

print("\nFEATURES AND TARGET")

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nFeature data types:")
print(X.dtypes)

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 9. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

# Get unique creation dates in chronological order
unique_dates = ml_df["created_date"].sort_values().unique()

# Select the date separating training and testing
split_date_index = int(len(unique_dates) * 0.80)
cutoff_date = unique_dates[split_date_index]

# Keep every task from the same date in the same dataset
train_mask = ml_df["created_date"] < cutoff_date
test_mask = ml_df["created_date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nCHRONOLOGICAL TRAIN / TEST SPLIT")

print("\nCutoff date:", cutoff_date)

print("\nTraining set:")
print("Shape:", X_train.shape)
print("Date range:",
      ml_df.loc[X_train.index, "created_date"].min(),
      "to",
      ml_df.loc[X_train.index, "created_date"].max())

print("\nTesting set:")
print("Shape:", X_test.shape)
print("Date range:",
      ml_df.loc[X_test.index, "created_date"].min(),
      "to",
      ml_df.loc[X_test.index, "created_date"].max())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Validate strict chronological separation
assert (
    ml_df.loc[X_train.index, "created_date"].max()
    <
    ml_df.loc[X_test.index, "created_date"].min()
)

print("\nChronological split validation passed!")
