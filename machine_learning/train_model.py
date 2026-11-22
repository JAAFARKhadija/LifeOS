import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD THE ML DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "synthetic" / "study_tasks_ml.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["created_date"])

print("Dataset loaded:", df.shape)

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

feature_columns = [
    "subject",
    "task_type",
    "priority",
    "estimated_duration",
    "days_until_deadline",
    "created_weekday"
]

X = df[feature_columns].copy()
y = df["completed_on_time"].copy()
# ==========================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

unique_dates = df["created_date"].sort_values().unique()

split_date_index = int(len(unique_dates) * 0.80)
cutoff_date = unique_dates[split_date_index]

train_mask = df["created_date"] < cutoff_date
test_mask = df["created_date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Cutoff date:", cutoff_date)

# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

categorical_features = [
    "subject",
    "task_type",
    "priority"
]

numerical_features = [
    "estimated_duration",
    "days_until_deadline",
    "created_weekday"
]

# ==========================================
# 5. CREATE PREPROCESSING PIPELINE
# ==========================================

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_features),
        ("numerical", numerical_transformer, numerical_features)
    ]
)

# ==========================================
# 5.1. RANDOM FOREST PREPROCESSING
# ==========================================

rf_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)

X_train_rf = rf_preprocessor.fit_transform(X_train)
X_test_rf = rf_preprocessor.transform(X_test)

print("\nRANDOM FOREST PREPROCESSING")
print("Training shape:", X_train_rf.shape)
print("Testing shape:", X_test_rf.shape)

# ==========================================
# 6. PREPROCESS TRAINING AND TESTING DATA
# ==========================================

# Learn transformations from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to testing data
X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\nPREPROCESSING RESULTS")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nTransformed feature names:")

feature_names = preprocessor.get_feature_names_out()

print(feature_names)

print("\nPreprocessing completed successfully!")

# ==========================================
# 8. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_processed, y_train)

print("\nMODEL TRAINING")
print("Model:", model)
print("Training completed successfully!")

train_accuracy = model.score(
    X_train_processed,
    y_train
)

print("Training accuracy:", round(train_accuracy, 4))


# ==========================================
# 8.1. MODEL COEFFICIENT ANALYSIS
# ==========================================

feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_[0]

coefficient_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

coefficient_df["odds_ratio"] = np.exp(
    coefficient_df["coefficient"]
)

coefficient_df = coefficient_df.sort_values(
    by="coefficient",
    ascending=False
)

print("\nMODEL COEFFICIENT ANALYSIS")
print(coefficient_df.to_string(index=False))


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

# Predict classes
y_pred = model.predict(X_test_processed)

# Predict probabilities for class 1
y_proba = model.predict_proba(X_test_processed)[:, 1]

# Calculate evaluation metrics
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nMODEL EVALUATION")
print("Test Accuracy:", round(test_accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Late / Unfinished", "Completed on time"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 10. BASELINE MODEL COMPARISON
# ==========================================

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

# Train the baseline
baseline_model.fit(X_train_processed, y_train)

# Predict testing data
baseline_predictions = baseline_model.predict(
    X_test_processed
)

# Evaluate baseline
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBASELINE MODEL COMPARISON")

print(
    "Logistic Regression accuracy:",
    round(test_accuracy, 4)
)

print(
    "Baseline accuracy:",
    round(baseline_accuracy, 4)
)

print(
    "Improvement:",
    round(
        (test_accuracy - baseline_accuracy) * 100,
        2
    ),
    "percentage points"
)

# ==========================================
# 10.1. RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_rf, y_train)

# Predictions
rf_pred = rf_model.predict(X_test_rf)
rf_proba = rf_model.predict_proba(X_test_rf)[:, 1]

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_roc_auc = roc_auc_score(y_test, rf_proba)

print("\nRANDOM FOREST EVALUATION")
print("Training accuracy:", round(rf_model.score(X_train_rf, y_train), 4))
print("Test accuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1-score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_roc_auc, 4))

print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Late / Unfinished", "Completed on time"],
    zero_division=0
))

print("\nRANDOM FOREST CONFUSION MATRIX")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 10.2. MODEL COMPARISON
# ==========================================

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Logistic Regression": [
        test_accuracy, precision, recall, f1, roc_auc
    ],
    "Random Forest": [
        rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc
    ]
})

print("\nMODEL COMPARISON")
print(comparison_df.to_string(index=False))

# ==========================================
# 10.3. SAVE THE SELECTED MODEL
# ==========================================

MODEL_DIR = BASE_DIR / "machine_learning" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "random_forest_study_model.joblib"

model_bundle = {
    "model": rf_model,
    "preprocessor": rf_preprocessor,
    "feature_columns": feature_columns
}

joblib.dump(model_bundle, model_path)

print("\nMODEL SAVED SUCCESSFULLY")
print("Model path:", model_path)

# ==========================================
# 11. TARGET ANALYSIS BY MONTH
# ==========================================

# Load creation month and target together
analysis_df = df[["created_month", "completed_on_time"]].copy()

monthly_results = (
    analysis_df
    .groupby("created_month")["completed_on_time"]
    .agg(["count", "mean"])
)

monthly_results["on_time_percentage"] = (
    monthly_results["mean"] * 100
).round(2)

print("\nON-TIME COMPLETION BY MONTH")

print(monthly_results)


# ==========================================
# 12. FEATURE RELATIONSHIP ANALYSIS
# ==========================================

# Analyze training data only
analysis_df = df.loc[
    train_mask,
    ["priority", "estimated_duration", "completed_on_time"]
].copy()

# 1. On-time rate by priority
print("\nON-TIME RATE BY PRIORITY")

priority_analysis = (
    analysis_df
    .groupby("priority")["completed_on_time"]
    .agg(["count", "mean"])
)

priority_analysis["on_time_percentage"] = (
    priority_analysis["mean"] * 100
).round(2)

print(priority_analysis)


# 2. On-time rate by task duration
analysis_df["duration_group"] = pd.cut(
    analysis_df["estimated_duration"],
    bins=[0, 240, float("inf")],
    labels=["240 minutes or less", "More than 240 minutes"]
)

print("\nON-TIME RATE BY DURATION")

duration_analysis = (
    analysis_df
    .groupby("duration_group", observed=False)["completed_on_time"]
    .agg(["count", "mean"])
)

duration_analysis["on_time_percentage"] = (
    duration_analysis["mean"] * 100
).round(2)

print(duration_analysis)
