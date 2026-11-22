
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "machine_learning"
    / "models"
    / "random_forest_study_model.joblib"
)

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
feature_columns = model_bundle["feature_columns"]


# ==========================================
# 2. PREDICT TASK COMPLETION
# ==========================================

def predict_task_completion(
    subject,
    task_type,
    priority,
    estimated_duration,
    created_date,
    due_date
):

    if estimated_duration is None:
        raise ValueError("Task duration is missing.")

    created = pd.to_datetime(created_date, errors="coerce")
    deadline = pd.to_datetime(due_date, errors="coerce")

    if pd.isna(created) or pd.isna(deadline):
        raise ValueError("Invalid task creation date or deadline.")

    days_until_deadline = (deadline - created).days
    created_weekday = created.weekday()

    task_data = pd.DataFrame([{
        "subject": subject if subject else "Unknown",
        "task_type": task_type,
        "priority": priority,
        "estimated_duration": float(estimated_duration),
        "days_until_deadline": days_until_deadline,
        "created_weekday": created_weekday
    }])

    task_data = task_data[feature_columns]

    processed_data = preprocessor.transform(task_data)

    prediction = int(model.predict(processed_data)[0])

    probabilities = model.predict_proba(processed_data)[0]

    class_index = list(model.classes_).index(1)
    on_time_probability = float(probabilities[class_index])

    return {
        "prediction": prediction,
        "on_time_probability": on_time_probability,
        "late_probability": 1 - on_time_probability
    }
