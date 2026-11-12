
from database.db import get_connection


# 1. Create a new habit
def add_habit(name, description=None):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO habits (name, description)
        VALUES (?, ?)
    """, (name, description))

    connection.commit()
    connection.close()

    print(f"Habit '{name}' added successfully!")


# 2. Retrieve all habits
def get_all_habits():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM habits")

    habits = cursor.fetchall()
    connection.close()

    return habits


# 3. Record daily habit completion
def log_habit(habit_id, log_date, completed=True):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO habit_logs (habit_id, log_date, completed)
        VALUES (?, ?, ?)
        ON CONFLICT(habit_id, log_date)
        DO UPDATE SET completed = excluded.completed
    """, (habit_id, log_date, int(completed)))

    connection.commit()
    connection.close()

    print(f"Habit {habit_id} logged successfully!")


# 4. Retrieve habit history
def get_habit_logs(habit_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM habit_logs
        WHERE habit_id = ?
        ORDER BY log_date DESC
    """, (habit_id,))

    logs = cursor.fetchall()
    connection.close()

    return logs
