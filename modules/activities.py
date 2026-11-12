
from database.db import get_connection


def add_activity(activity_type, title, duration,
                 activity_date, notes=None):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO activities (
            activity_type,
            title,
            duration,
            activity_date,
            notes
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        activity_type,
        title,
        duration,
        activity_date,
        notes
    ))

    connection.commit()
    connection.close()

    print(f"{activity_type} activity added successfully!")


def get_all_activities():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM activities
        ORDER BY activity_date DESC
    """)

    activities = cursor.fetchall()
    connection.close()

    return activities


def get_activities_by_type(activity_type):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM activities
        WHERE activity_type = ?
        ORDER BY activity_date DESC
    """, (activity_type,))

    activities = cursor.fetchall()
    connection.close()

    return activities
