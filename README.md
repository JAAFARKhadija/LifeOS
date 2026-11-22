# 🌱 LifeOS – AI-Powered Personal Life Assistant

## 1. Project Overview

LifeOS is an AI-powered personal productivity application developed with Python and Streamlit. Its purpose is to help users organize and monitor different aspects of their daily lives through a centralized dashboard.

The application combines task management, activity tracking, habit monitoring, Machine Learning predictions, and an AI assistant to provide personalized productivity support.

## 2. Project Objectives

The main objectives of LifeOS are to:

- Organize academic tasks, deadlines, and priorities.
- Track study sessions, gym workouts, and volleyball activities.
- Monitor daily habits and personal routines.
- Use Machine Learning to estimate the likelihood of completing study tasks on time.
- Integrate an AI assistant capable of answering questions using application data.
- Provide automatic deadline reminders and daily productivity summaries.

## 3. Main Features

### 3.1 Personal Dashboard
- Display an overview of study tasks.
- Show completed and pending tasks.
- Display activity statistics.
- Visualize weekly activity.

### 3.2 Study Task Management
- Add, update, complete, and delete study tasks.
- Organize tasks by subject and task type.
- Set deadlines, priorities, and estimated durations.
- Monitor pending and completed tasks.

### 3.3 Activities Management
Track three types of activities:
- Study
- Gym
- Volleyball

Record activity dates, durations, titles, and notes.

### 3.4 Daily Habit Tracker
- Create and manage personal habits.
- Record daily habit completion.
- Monitor habit progress.

### 3.5 Machine Learning – Task Completion Prediction

LifeOS integrates a Machine Learning model that predicts whether a study task is likely to be completed on time.

**Model:** Random Forest Classifier

**Input features:**
- Subject
- Task type
- Priority
- Estimated duration
- Days until deadline
- Creation weekday

**Target variable:** `completed_on_time`

The model was trained using a synthetic study-task dataset.

Reported test performance:

| Metric | Score |
|---|---:|
| Accuracy | 50.00% |
| Precision | 57.45% |
| Recall | 48.21% |
| F1-score | 52.43% |
| ROC-AUC | 56.68% |

The model is experimental. Its performance is limited, and its predictions should be interpreted as estimates rather than reliable guarantees.

### 3.6 AI Assistant – Gemini Integration

LifeOS includes an AI assistant powered by Google's Gemini API.

The assistant can:
- Answer questions about pending tasks and deadlines.
- Summarize recent activities.
- Analyze habit progress.
- Provide productivity suggestions.
- Generate daily productivity reports.

The assistant uses read-only functions to retrieve relevant information from the SQLite database.

### 3.7 Automation

LifeOS includes:
- Automatic dashboard reminders for overdue tasks, tasks due today, and upcoming deadlines.
- An on-demand daily productivity summary generated through the AI assistant.

The daily summary is triggered by the user through a button in the application.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| SQLite | Database management |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Plotly | Data visualization |
| Joblib | Saving and loading the ML model |
| Google Gemini API | AI assistant |
| python-dotenv | Environment variable management |

## 5. Project Structure

```text
ProjetML_LifeOS_JAAFARKhadija/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── database/
│   ├── db.py
│   └── init_db.py
│
├── modules/
│   ├── tasks.py
│   ├── activities.py
│   ├── habits.py
│   └── ai_agent.py
│
├── machine_learning/
│   ├── models/
│   │   └── random_forest_study_model.joblib
│   ├── train_model.py
│   └── predict.py
│
└── data/
    └── synthetic/
        ├── study_tasks.csv
        ├── study_sessions.csv
        ├── gym_sessions.csv
        ├── volleyball_sessions.csv
        ├── habit_logs.csv
        └── daily_summary.csv
```

*Note: The structure above highlights the main project files. Additional scripts and files may also exist in the repository.*

## 6. Installation and Setup

### Prerequisites
- Python 3.11 or later
- pip
- A Google Gemini API key

### Step 1: Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ProjetML_LifeOS_JAAFARKhadija
```

### Step 2: Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key.

**Important:** Never commit your `.env` file or expose your API key.

### Step 5: Initialize the database

If the database has not already been initialized, run:

```bash
python -m database.init_db
```

### Step 6: Launch LifeOS

```bash
streamlit run app.py
```

The application will open in your browser.

## 7. Synthetic Data

Synthetic datasets were generated to simulate personal productivity activities over a one-year period.

They include:
- Study tasks and their completion outcomes.
- Study sessions.
- Gym sessions.
- Volleyball sessions.
- Daily habit logs.
- Daily productivity summaries.

These datasets were created for development, testing, and Machine Learning experimentation. They do not represent verified real-life activity records.

## 8. Limitations

- The Machine Learning model has limited predictive performance because it was trained on synthetic data.
- Predictions are experimental and should not be treated as certain outcomes.
- The AI assistant requires an internet connection and a valid Gemini API key.
- AI responses may occasionally be inaccurate.
- Daily productivity summaries are generated on demand rather than through a background scheduling service.

## 9. Future Improvements

Potential future developments include:
- Collecting more real-world data to improve model performance.
- Experimenting with additional Machine Learning algorithms.
- Improving prediction accuracy and evaluation.
- Adding more advanced productivity analytics.
- Implementing background scheduled notifications.
- Improving AI recommendations and personalization.
- Developing a more advanced user interface.

## 10. Conclusion

LifeOS demonstrates how Python, data processing, Machine Learning, AI Agents, and automation can be combined in a practical productivity application.

The project provides a centralized environment for managing daily activities, tracking habits, organizing academic work, and receiving AI-powered productivity assistance.
