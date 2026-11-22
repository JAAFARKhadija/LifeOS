
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from database.db import get_connection


# ==========================================
# 1. LOAD GEMINI API KEY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from the .env file.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash"


# ==========================================
# 2. DATABASE TOOLS
# ==========================================

def get_tasks():
    """Retrieve the user's study tasks, deadlines, priorities and statuses."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id, title, subject, task_type,
            due_date, priority, status, estimated_duration
        FROM tasks
        ORDER BY due_date ASC
    """)

    rows = cursor.fetchall()
    connection.close()

    tasks = [
        {
            "id": row[0],
            "title": row[1],
            "subject": row[2],
            "task_type": row[3],
            "deadline": row[4],
            "priority": row[5],
            "status": row[6],
            "duration_minutes": row[7]
        }
        for row in rows
    ]

    return json.dumps(tasks, ensure_ascii=False)


def get_recent_activities():
    """Retrieve activities recorded during the last seven days."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            activity_type, title, duration, activity_date
        FROM activities
        WHERE DATE(activity_date) >= DATE('now', '-7 days')
        ORDER BY activity_date DESC
    """)

    rows = cursor.fetchall()
    connection.close()

    activities = [
        {
            "type": row[0],
            "title": row[1],
            "duration_hours": row[2],
            "date": row[3]
        }
        for row in rows
    ]

    return json.dumps(activities, ensure_ascii=False)



def get_habit_progress():
    """Retrieve habit completion statistics for the last 30 days."""

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            habit_id,
            COUNT(*) AS total_logs,
            SUM(completed) AS completed_logs
        FROM habit_logs
        WHERE DATE(log_date) >= DATE('now', '-29 days')
        GROUP BY habit_id
        ORDER BY habit_id
    """)

    rows = cursor.fetchall()
    connection.close()

    progress = [
        {
            "habit": f"Habit {index + 1}",
            "total_logs": row[1],
            "completed_logs": row[2],
            "completion_rate": (
                round(row[2] / row[1] * 100, 1)
                if row[1] else 0
            )
        }
        for index, row in enumerate(rows)
    ]

    return json.dumps(progress, ensure_ascii=False)

# ==========================================
# 3. GEMINI AI AGENT - INTERACTIONS API
# ==========================================

SYSTEM_INSTRUCTION = """
You are LifeOS AI Assistant, a personal productivity assistant.

Your responsibilities:
- Help organize study tasks and deadlines.
- Analyze study, gym and volleyball activities.
- Review general daily habit progress.
- Suggest realistic daily study plans.
- Give practical productivity advice.

You have three read-only tools:
get_tasks, get_recent_activities and get_habit_progress.

Rules:
- Use tools when questions require actual LifeOS data.
- Never invent tasks, activities or statistics.
- Distinguish database facts from suggestions.
- Give clear, concise and actionable answers.
- Never claim to create, edit or delete records.
- Treat ML predictions as experimental estimates.
"""

TOOLS = [
    {
        "type": "function",
        "name": "get_tasks",
        "description": "Retrieve all study tasks, deadlines, priorities and statuses.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_recent_activities",
        "description": "Retrieve activities recorded during the last seven days.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_habit_progress",
        "description": "Retrieve recent habit completion statistics.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "get_tasks": get_tasks,
    "get_recent_activities": get_recent_activities,
    "get_habit_progress": get_habit_progress
}


def chat_with_lifeos(message, history=None):

    history = history or []

    conversation = ""

    for item in history[-8:]:
        conversation += (
            f"{item['role']}: {item['content']}\n"
        )

    prompt = f"""
{SYSTEM_INSTRUCTION}

Previous conversation:
{conversation}

User: {message}
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
        tools=TOOLS
    )

    # Handle tool calls
    for _ in range(5):

        function_results = []

        for step in interaction.steps or []:

            if step.type == "function_call":

                function_name = step.name

                if function_name not in AVAILABLE_FUNCTIONS:
                    continue

                function = AVAILABLE_FUNCTIONS[function_name]

                result = function(**(step.arguments or {}))

                function_results.append({
                    "type": "function_result",
                    "name": function_name,
                    "call_id": step.id,
                    "result": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                })

        if not function_results:
            break

        interaction = client.interactions.create(
            model=MODEL_NAME,
            previous_interaction_id=interaction.id,
            input=function_results
        )

    return interaction.output_text or "I couldn't generate a response."
