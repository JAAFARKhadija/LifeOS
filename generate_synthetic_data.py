from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "synthetic"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility: same random results every time
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Dataset period
START_DATE = "2025-11-01"
END_DATE = "2026-11-12"

# Lifestyle configuration
SUBJECTS = [
    "RO",
    "Big Data",
    "Proba",
    "ML",
    "Languages",
    "Data Warehouse",
    "Python",
    "TCF prepa",
    "Entretien Campus",
]

TASK_TYPES = [
    "Cours",
    "TD",
    "TP",
    "Projects",
    "Series",
    "Tests",
]

PRIORITIES = ["Low", "Medium", "High"]

print("LifeOS Synthetic Data Generator initialized.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Number of subjects: {len(SUBJECTS)}")
print(f"Number of task types: {len(TASK_TYPES)}")



# ==========================================
# STUDY TASK GENERATOR
# ==========================================

def generate_study_tasks():
    tasks = []

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    end_date = pd.Timestamp(END_DATE)

    task_id = 1

    for day in dates:
        month = day.month

        # Fewer tasks during summer
        if month in [6, 7, 8]:
            number_of_tasks = rng.integers(0, 2)
        else:
            number_of_tasks = rng.integers(1, 3)

        for _ in range(number_of_tasks):
            subject = rng.choice(SUBJECTS)
            task_type = rng.choice(TASK_TYPES)
            priority = rng.choice(
                PRIORITIES,
                p=[0.20, 0.50, 0.30]
            )

            # Estimated task duration in minutes
            duration = int(np.clip(
                rng.normal(150, 65), 30, 420
            ))

            # Deadline between 1 and 10 days after creation
            days_until_deadline = int(rng.integers(1, 11))
            deadline = day + pd.Timedelta(days=days_until_deadline)

            # Completion probability
            completion_probability = 0.72

            if priority == "High":
                completion_probability += 0.10

            if duration > 240:
                completion_probability -= 0.12

            if month in [6, 7, 8]:
                completion_probability -= 0.05

            completion_probability = np.clip(
                completion_probability, 0.15, 0.95
            )

            completed = rng.random() < completion_probability

            completion_date = None
            outcome = "Unfinished"

            if completed:
                delay = int(rng.choice(
                    [0, 1, 2, 3],
                    p=[0.65, 0.20, 0.10, 0.05]
                ))

                completion_date = (
                    deadline + pd.Timedelta(days=delay)
                )

                if completion_date > end_date:
                    outcome = "Pending"
                    completion_date = None
                elif delay == 0:
                    outcome = "Completed"
                else:
                    outcome = "Late"

            # A deadline that hasn't passed is still pending
            if deadline > end_date:
                outcome = "Pending"
                completion_date = None

            tasks.append({
                "task_id": task_id,
                "created_date": day.date(),
                "subject": subject,
                "task_type": task_type,
                "priority": priority,
                "estimated_duration": duration,
                "deadline": deadline.date(),
                "completion_date": (
                    completion_date.date()
                    if completion_date is not None
                    else None
                ),
                "outcome": outcome,
                "is_synthetic": True
            })

            task_id += 1

    df = pd.DataFrame(tasks)

    output_path = OUTPUT_DIR / "study_tasks.csv"
    df.to_csv(output_path, index=False)

    print("\nStudy tasks generated successfully!")
    print(f"Total tasks: {len(df)}")
    print("\nOutcome distribution:")
    print(df["outcome"].value_counts())
    print(f"\nSaved to: {output_path}")

    return df


# Generate dataset
study_tasks_df = generate_study_tasks()


# ==========================================
# STUDY SESSION GENERATOR
# ==========================================

def generate_study_sessions():
    sessions = []
    session_id = 1

    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    for day in dates:
        month = day.month

        # Daily study duration in minutes
        if month in [6, 7, 8]:
            total_minutes = int(rng.integers(60, 181))
            number_of_sessions = int(rng.integers(1, 3))
        else:
            total_minutes = int(rng.integers(240, 481))
            number_of_sessions = int(rng.integers(1, 4))

        # At least 45 minutes per session
        number_of_sessions = min(
            number_of_sessions,
            total_minutes // 45
        )

        remaining = total_minutes - (45 * number_of_sessions)

        # Distribute the remaining time across sessions
        weights = rng.dirichlet(
            np.ones(number_of_sessions)
        )

        extra_minutes = np.floor(
            weights * remaining
        ).astype(int)

        durations = extra_minutes + 45

        # Correct rounding difference
        durations[-1] += total_minutes - durations.sum()

        # Starting time between 08:00 and 11:00
        start_hour = int(rng.integers(8, 12))
        current_minutes = start_hour * 60

        for duration in durations:
            start_time = (
                f"{current_minutes // 60:02d}:"
                f"{current_minutes % 60:02d}"
            )

            sessions.append({
                "session_id": session_id,
                "study_date": day.date(),
                "subject": rng.choice(SUBJECTS),
                "start_time": start_time,
                "duration_minutes": int(duration),
                "is_synthetic": True
            })

            session_id += 1

            # 30-minute break between sessions
            current_minutes += int(duration) + 30

    df = pd.DataFrame(sessions)

    output_path = OUTPUT_DIR / "study_sessions.csv"
    df.to_csv(output_path, index=False)

    print("\nStudy sessions generated successfully!")
    print(f"Total sessions: {len(df)}")
    print(
        f"Total study hours: "
        f"{df['duration_minutes'].sum() / 60:.1f}"
    )
    print(f"Saved to: {output_path}")

    return df


# Generate study sessions
study_sessions_df = generate_study_sessions()

# ==========================================
# GYM SESSION GENERATOR
# ==========================================


def generate_gym_sessions():
    sessions = []
    session_id = 1
    workout_types = ["Push", "Pull", "Legs"]
    workout_index = 0

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    weeks = pd.Series(dates, index=dates).groupby(
        dates.to_period("W-SUN")
    )

    for _, week_dates in weeks:
        available_dates = [
            day for day in week_dates
            if day.month not in [6, 7, 8]
        ]

        # Separate November 2026 from the normal routine
        special_dates = [
            day for day in available_dates
            if day.year == 2026 and day.month == 11
            and day.weekday() != 4
        ]

        normal_dates = [
            day for day in available_dates
            if not (day.year == 2026 and day.month == 11)
        ]

        # Normal routine: 4–5 sessions per week
        if normal_dates:
            number_of_sessions = int(rng.integers(4, 6))
            number_of_sessions = min(
                number_of_sessions, len(normal_dates)
            )

            selected_indices = rng.choice(
                len(normal_dates),
                size=number_of_sessions,
                replace=False
            )

            gym_dates = [
                normal_dates[i] for i in sorted(selected_indices)
            ]
        else:
            gym_dates = []

        # November 2026: every day except Friday
        gym_dates.extend(special_dates)
        gym_dates.sort()

        for day in gym_dates:
            workout_type = workout_types[
                workout_index % len(workout_types)
            ]

            duration = int(rng.integers(60, 91))

            sessions.append({
                "session_id": session_id,
                "gym_date": day.date(),
                "workout_type": workout_type,
                "duration_minutes": duration,
                "is_synthetic": True
            })

            session_id += 1
            workout_index += 1

    df = pd.DataFrame(sessions)

    output_path = OUTPUT_DIR / "gym_sessions.csv"
    df.to_csv(output_path, index=False)

    print("\nGym sessions generated successfully!")
    print(f"Total sessions: {len(df)}")
    print("\nWorkout distribution:")
    print(df["workout_type"].value_counts())
    print(f"\nSaved to: {output_path}")

    return df


gym_sessions_df = generate_gym_sessions()


# ==========================================
# VOLLEYBALL SESSION GENERATOR
# ==========================================

def generate_volleyball_sessions():
    sessions = []
    session_id = 1

    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    weeks = pd.Series(dates, index=dates).groupby(
        dates.to_period("W-SUN")
    )

    for _, week_dates in weeks:
        # Exclude summer
        available_dates = [
            day for day in week_dates
            if day.month not in [6, 7, 8]
        ]

        if len(available_dates) == 0:
            continue

        # Normal routine: 2 or 3 sessions per week
        number_of_sessions = int(rng.integers(2, 4))
        number_of_sessions = min(
            number_of_sessions,
            len(available_dates)
        )

        selected_indices = rng.choice(
            len(available_dates),
            size=number_of_sessions,
            replace=False
        )

        volleyball_dates = [
            available_dates[i]
            for i in sorted(selected_indices)
        ]

        for day in volleyball_dates:
            duration = int(rng.integers(120, 181))

            sessions.append({
                "session_id": session_id,
                "volleyball_date": day.date(),
                "duration_minutes": duration,
                "is_synthetic": True
            })

            session_id += 1

    df = pd.DataFrame(sessions)

    output_path = OUTPUT_DIR / "volleyball_sessions.csv"
    df.to_csv(output_path, index=False)

    print("\nVolleyball sessions generated successfully!")
    print(f"Total sessions: {len(df)}")
    print(
        f"Total volleyball hours: "
        f"{df['duration_minutes'].sum() / 60:.1f}"
    )
    print(f"Saved to: {output_path}")

    return df


# Generate volleyball sessions
volleyball_sessions_df = generate_volleyball_sessions()


# ==========================================
# HABIT LOG GENERATOR
# ==========================================

def generate_habit_logs():
    logs = []
    log_id = 1

    checkbox_habits = {
        "Quran": 0.75,
        "Fajr": 0.85,
        "Dhuhr": 0.85,
        "Asr": 0.85,
        "Maghrib": 0.85,
        "Isha": 0.85,
        "Face Icing": 0.45,
        "French Practice": 0.50
    }

    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    for day in dates:

        # Generate checkbox habits
        for habit, probability in checkbox_habits.items():
            completed = int(rng.random() < probability)

            logs.append({
                "log_id": log_id,
                "log_date": day.date(),
                "habit": habit,
                "completed": completed,
                "numeric_value": None,
                "value_type": "boolean",
                "is_synthetic": True
            })

            log_id += 1

        # Generate numeric sleep hours
        sleep_hours = float(
            np.clip(rng.normal(7.2, 1.0), 5.5, 9.5)
        )

        sleep_hours = round(sleep_hours * 4) / 4

        logs.append({
            "log_id": log_id,
            "log_date": day.date(),
            "habit": "Sleep",
            "completed": None,
            "numeric_value": sleep_hours,
            "value_type": "numeric",
            "is_synthetic": True
        })

        log_id += 1

    df = pd.DataFrame(logs)

    output_path = OUTPUT_DIR / "habit_logs.csv"
    df.to_csv(output_path, index=False)

    print("\nHabit logs generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Unique habits: {df['habit'].nunique()}")
    print(f"Date range: {df['log_date'].min()} to {df['log_date'].max()}")

    print("\nCheckbox completion rates:")
    checkbox_summary = (
        df[df["value_type"] == "boolean"]
        .groupby("habit")["completed"]
        .mean() * 100
    )
    print(checkbox_summary.round(1))

    print("\nSleep statistics:")
    sleep = df[df["habit"] == "Sleep"]["numeric_value"]
    print(sleep.describe().round(2))

    print(f"\nSaved to: {output_path}")

    return df


# Generate habit logs
habit_logs_df = generate_habit_logs()


# ==========================================
# DAILY SUMMARY GENERATOR
# ==========================================

def generate_daily_summary():

    # Create one row for every date
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    daily = pd.DataFrame({"date": dates})

    # --------------------------------------
    # 1. Study sessions
    # --------------------------------------

    study = pd.read_csv(
        OUTPUT_DIR / "study_sessions.csv",
        parse_dates=["study_date"]
    )

    study_daily = study.groupby("study_date").agg(
        study_sessions=("session_id", "count"),
        study_minutes=("duration_minutes", "sum")
    ).reset_index()

    study_daily = study_daily.rename(
        columns={"study_date": "date"}
    )

    daily = daily.merge(study_daily, on="date", how="left")

    # --------------------------------------
    # 2. Gym sessions
    # --------------------------------------

    gym = pd.read_csv(
        OUTPUT_DIR / "gym_sessions.csv",
        parse_dates=["gym_date"]
    )

    gym_daily = gym.groupby("gym_date").agg(
        gym_sessions=("session_id", "count"),
        gym_minutes=("duration_minutes", "sum")
    ).reset_index()

    gym_daily = gym_daily.rename(
        columns={"gym_date": "date"}
    )

    daily = daily.merge(gym_daily, on="date", how="left")

    # --------------------------------------
    # 3. Volleyball sessions
    # --------------------------------------

    volleyball = pd.read_csv(
        OUTPUT_DIR / "volleyball_sessions.csv",
        parse_dates=["volleyball_date"]
    )

    volleyball_daily = volleyball.groupby("volleyball_date").agg(
        volleyball_sessions=("session_id", "count"),
        volleyball_minutes=("duration_minutes", "sum")
    ).reset_index()

    volleyball_daily = volleyball_daily.rename(
        columns={"volleyball_date": "date"}
    )

    daily = daily.merge(
        volleyball_daily, on="date", how="left"
    )

    # --------------------------------------
    # 4. Habits
    # --------------------------------------

    habits = pd.read_csv(
        OUTPUT_DIR / "habit_logs.csv",
        parse_dates=["log_date"]
    )

    # Checkbox habits
    checkbox = habits[
        habits["value_type"] == "boolean"
    ]

    checkbox_daily = checkbox.pivot(
        index="log_date",
        columns="habit",
        values="completed"
    ).reset_index()

    checkbox_daily = checkbox_daily.rename(
        columns={
            "log_date": "date",
            "Quran": "quran_completed",
            "Fajr": "fajr_completed",
            "Dhuhr": "dhuhr_completed",
            "Asr": "asr_completed",
            "Maghrib": "maghrib_completed",
            "Isha": "isha_completed",
            "Face Icing": "face_icing_completed",
            "French Practice": "french_completed"
        }
    )

    daily = daily.merge(
        checkbox_daily, on="date", how="left"
    )

    # Numeric sleep values
    sleep_daily = habits[
        habits["habit"] == "Sleep"
    ][["log_date", "numeric_value"]].rename(
        columns={
            "log_date": "date",
            "numeric_value": "sleep_hours"
        }
    )

    daily = daily.merge(
        sleep_daily, on="date", how="left"
    )

    # Total prayers completed each day
    prayer_columns = [
        "fajr_completed",
        "dhuhr_completed",
        "asr_completed",
        "maghrib_completed",
        "isha_completed"
    ]

    daily["prayers_completed"] = daily[
        prayer_columns
    ].sum(axis=1)

    # --------------------------------------
    # 5. Study task activity
    # --------------------------------------

    tasks = pd.read_csv(
        OUTPUT_DIR / "study_tasks.csv",
        parse_dates=["created_date", "completion_date"]
    )

    created_daily = tasks.groupby("created_date").size().rename(
        "tasks_created"
    ).reset_index().rename(
        columns={"created_date": "date"}
    )

    completed_daily = (
        tasks.dropna(subset=["completion_date"])
        .groupby("completion_date")
        .size()
        .rename("tasks_completed")
        .reset_index()
        .rename(columns={"completion_date": "date"})
    )

    daily = daily.merge(
        created_daily, on="date", how="left"
    )

    daily = daily.merge(
        completed_daily, on="date", how="left"
    )

    # --------------------------------------
    # 6. Clean and calculate features
    # --------------------------------------

    count_columns = [
        "study_sessions",
        "study_minutes",
        "gym_sessions",
        "gym_minutes",
        "volleyball_sessions",
        "volleyball_minutes",
        "tasks_created",
        "tasks_completed"
    ]

    daily[count_columns] = daily[count_columns].fillna(0)

    daily["study_hours"] = (
        daily["study_minutes"] / 60
    ).round(2)

    daily["is_synthetic"] = True

    # --------------------------------------
    # 7. Save dataset
    # --------------------------------------

    output_path = OUTPUT_DIR / "daily_summary.csv"

    daily.to_csv(output_path, index=False)

    print("\nDaily summary generated successfully!")
    print(f"Total days: {len(daily)}")
    print(f"Total columns: {len(daily.columns)}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    print("\nDaily summary preview:")
    print(daily.head())

    print(f"\nSaved to: {output_path}")

    return daily

# Generate daily summary
daily_summary_df = generate_daily_summary()
