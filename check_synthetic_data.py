
import pandas as pd
from pathlib import Path

file_path = Path("data/synthetic/study_tasks.csv")

df = pd.read_csv(file_path)

print("DATASET SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE TASK IDs:")
print(df["task_id"].duplicated().sum())

print("\nTASK TYPES:")
print(df["task_type"].value_counts())

print("\nSUBJECTS:")
print(df["subject"].value_counts())

print("\nOUTCOME COUNTS:")
print(df["outcome"].value_counts())


# ==========================================
# STUDY SESSION VALIDATION
# ==========================================

sessions = pd.read_csv(
    "data/synthetic/study_sessions.csv"
)

sessions["study_date"] = pd.to_datetime(
    sessions["study_date"]
)

daily_hours = (
    sessions.groupby("study_date")["duration_minutes"]
    .sum() / 60
)

summer = daily_hours[
    daily_hours.index.month.isin([6, 7, 8])
]

regular = daily_hours[
    ~daily_hours.index.month.isin([6, 7, 8])
]

print("\nDAILY STUDY HOURS VALIDATION")

print("\nSummer:")
print(f"Minimum: {summer.min():.2f} hours")
print(f"Maximum: {summer.max():.2f} hours")
print(f"Invalid days: {(~summer.between(1, 3)).sum()}")

print("\nRest of the year:")
print(f"Minimum: {regular.min():.2f} hours")
print(f"Maximum: {regular.max():.2f} hours")
print(f"Invalid days: {(~regular.between(4, 8)).sum()}")

# ==========================================
# GYM DATA VALIDATION
# ==========================================

gym = pd.read_csv("data/synthetic/gym_sessions.csv")
gym["gym_date"] = pd.to_datetime(gym["gym_date"])

print("\nGYM DATA VALIDATION")

# 1. Check session duration
print("\nSession duration:")
print(f"Minimum: {gym['duration_minutes'].min()} minutes")
print(f"Maximum: {gym['duration_minutes'].max()} minutes")
print(
    "Invalid sessions:",
    (~gym["duration_minutes"].between(60, 90)).sum()
)

# 2. Check summer sessions
summer_gym = gym[gym["gym_date"].dt.month.isin([6, 7, 8])]
print(f"\nSummer gym sessions: {len(summer_gym)}")

# 3. Check weekly session counts
gym["week"] = gym["gym_date"].dt.to_period("W-SUN")
weekly_counts = gym.groupby("week").size()

print("\nWeekly session counts:")
print(weekly_counts)

# 4. Check November 2026 exception
november_gym = gym[
    (gym["gym_date"].dt.year == 2026) &
    (gym["gym_date"].dt.month == 11)
]

print("\nNovember 2026:")
print(f"Total sessions: {len(november_gym)}")

print(
    "Friday sessions:",
    (november_gym["gym_date"].dt.weekday == 4).sum()
)


# ==========================================
# VOLLEYBALL DATA VALIDATION
# ==========================================

volleyball = pd.read_csv(
    "data/synthetic/volleyball_sessions.csv"
)

volleyball["volleyball_date"] = pd.to_datetime(
    volleyball["volleyball_date"]
)

print("\nVOLLEYBALL DATA VALIDATION")

# 1. Session duration
print("\nSession duration:")
print(f"Minimum: {volleyball['duration_minutes'].min()} minutes")
print(f"Maximum: {volleyball['duration_minutes'].max()} minutes")

print(
    "Invalid sessions:",
    (~volleyball["duration_minutes"].between(120, 180)).sum()
)

# 2. Summer sessions
summer_volleyball = volleyball[
    volleyball["volleyball_date"].dt.month.isin([6, 7, 8])
]

print(f"\nSummer volleyball sessions: {len(summer_volleyball)}")

# 3. Weekly frequency
volleyball["week"] = (
    volleyball["volleyball_date"].dt.to_period("W-SUN")
)

weekly_counts = volleyball.groupby("week").size()

print("\nWeekly session counts:")
print(weekly_counts)

# 4. Check complete weeks only
first_date = pd.Timestamp("2025-11-01")
last_date = pd.Timestamp("2026-11-12")

full_weeks = weekly_counts[
    (weekly_counts.index.start_time >= first_date) &
    (weekly_counts.index.end_time.normalize() <= last_date)
]

invalid_weeks = full_weeks[
    ~full_weeks.between(2, 3)
]

print("\nComplete-week validation:")
print(f"Complete weeks: {len(full_weeks)}")
print(f"Invalid weeks: {len(invalid_weeks)}")


# ==========================================
# HABIT DATA VALIDATION
# ==========================================

habits = pd.read_csv("data/synthetic/habit_logs.csv")
habits["log_date"] = pd.to_datetime(habits["log_date"])

print("\nHABIT DATA VALIDATION")

# 1. Dataset size and uniqueness
print(f"\nDataset shape: {habits.shape}")
print(f"Unique habits: {habits['habit'].nunique()}")
print(f"Duplicate log IDs: {habits['log_id'].duplicated().sum()}")

# 2. Records per day
daily_counts = habits.groupby("log_date").size()

print("\nRecords per day:")
print(f"Minimum: {daily_counts.min()}")
print(f"Maximum: {daily_counts.max()}")
print(f"Invalid days: {(daily_counts != 9).sum()}")

# 3. Check all expected habits
expected_habits = [
    "Quran", "Fajr", "Dhuhr", "Asr", "Maghrib",
    "Isha", "Face Icing", "French Practice", "Sleep"
]

print("\nMissing habits:")
print(set(expected_habits) - set(habits["habit"].unique()))

# 4. Validate checkbox values
checkbox = habits[habits["value_type"] == "boolean"]

invalid_checkbox = (
    ~checkbox["completed"].isin([0, 1])
).sum()

print("\nCheckbox validation:")
print(f"Invalid checkbox values: {invalid_checkbox}")
print(f"Unexpected numeric values: {checkbox['numeric_value'].notna().sum()}")

# 5. Validate sleep values
sleep = habits[habits["habit"] == "Sleep"]

print("\nSleep validation:")
print(f"Minimum: {sleep['numeric_value'].min()} hours")
print(f"Maximum: {sleep['numeric_value'].max()} hours")
print(f"Missing sleep values: {sleep['numeric_value'].isna().sum()}")
print(
    "Invalid sleep values:",
    (~sleep["numeric_value"].between(5.5, 9.5)).sum()
)

# 6. Check synthetic labels
print("\nSynthetic data validation:")
print(f"Non-synthetic records: {(habits['is_synthetic'] != True).sum()}")

# ==========================================
# DAILY SUMMARY VALIDATION
# ==========================================

daily = pd.read_csv(
    "data/synthetic/daily_summary.csv",
    parse_dates=["date"]
)

study = pd.read_csv("data/synthetic/study_sessions.csv")
gym = pd.read_csv("data/synthetic/gym_sessions.csv")
volleyball = pd.read_csv("data/synthetic/volleyball_sessions.csv")
tasks = pd.read_csv("data/synthetic/study_tasks.csv")

print("\n--- DAILY SUMMARY VALIDATION ---")

# 1. Structure
print("Shape:", daily.shape)
assert len(daily) == 377
assert daily["date"].is_unique
assert daily["date"].is_monotonic_increasing
assert daily.isnull().sum().sum() == 0

# 2. Date continuity
expected_dates = pd.date_range("2025-11-01", "2026-11-12")
assert daily["date"].tolist() == expected_dates.tolist()

# 3. Study hours
summer = daily["date"].dt.month.isin([6, 7, 8])

assert daily.loc[summer, "study_hours"].between(1, 3).all()
assert daily.loc[~summer, "study_hours"].between(4, 8).all()

# 4. Gym and volleyball
assert (daily.loc[summer, "gym_sessions"] == 0).all()
assert (daily.loc[summer, "volleyball_sessions"] == 0).all()

assert daily["gym_minutes"].between(
    daily["gym_sessions"] * 60,
    daily["gym_sessions"] * 90
).all()

assert daily["volleyball_minutes"].between(
    daily["volleyball_sessions"] * 120,
    daily["volleyball_sessions"] * 180
).all()

# 5. Habits
habit_columns = [
    "quran_completed", "fajr_completed", "dhuhr_completed",
    "asr_completed", "maghrib_completed", "isha_completed",
    "face_icing_completed", "french_completed"
]

for column in habit_columns:
    assert daily[column].isin([0, 1]).all()

assert daily["prayers_completed"].between(0, 5).all()
assert daily["sleep_hours"].between(5.5, 9.5).all()

# 6. Synthetic data flag
assert daily["is_synthetic"].astype(str).str.lower().eq("true").all()

# 7. Cross-check totals against original datasets
assert daily["study_minutes"].sum() == study["duration_minutes"].sum()
assert daily["gym_minutes"].sum() == gym["duration_minutes"].sum()
assert daily["volleyball_minutes"].sum() == volleyball["duration_minutes"].sum()
assert daily["tasks_created"].sum() == len(tasks)

print("All daily summary validation checks passed!")
print("Daily records:", len(daily))
print("Columns:", len(daily.columns))
print("Total study hours:", round(daily["study_hours"].sum(), 2))
print("Total gym sessions:", daily["gym_sessions"].sum())
print("Total volleyball sessions:", daily["volleyball_sessions"].sum())
print("Total tasks created:", daily["tasks_created"].sum())