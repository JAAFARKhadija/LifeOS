
from database.db import get_connection


def add_task(title, subject=None, due_date=None,
             priority="Medium", estimated_duration=None):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO tasks (
            title,
            subject,
            due_date,
            priority,
            estimated_duration
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        title,
        subject,
        due_date,
        priority,
        estimated_duration
    ))

    connection.commit()
    connection.close()

    print(f"Task '{title}' added successfully!")


def get_all_tasks():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM tasks
        ORDER BY due_date ASC
    """)

    tasks = cursor.fetchall()

    connection.close()

    return tasks


def update_task_status(task_id, new_status):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE tasks
        SET status = ?
        WHERE id = ?
    """, (new_status, task_id))

    connection.commit()
    connection.close()

    print(f"Task {task_id} status updated to {new_status}!")


def delete_task(task_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM tasks
        WHERE id = ?
    """, (task_id,))

    connection.commit()
    connection.close()

    print(f"Task {task_id} deleted successfully!")


def get_task_by_id(task_id):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM tasks
        WHERE id = ?
    """, (task_id,))

    task = cursor.fetchone()

    connection.close()

    return task
