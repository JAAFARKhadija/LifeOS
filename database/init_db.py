
from database.db import get_connection


def create_tables():

    connection = get_connection()
    cursor = connection.cursor()

    # 1. Study tasks
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            subject TEXT,
            task_type TEXT DEFAULT 'Cours',
            created_date TEXT,
            due_date TEXT,
            priority TEXT DEFAULT 'Medium'
                CHECK(priority IN ('Low', 'Medium', 'High')),
            status TEXT DEFAULT 'Pending'
                CHECK(status IN ('Pending', 'Completed')),
            estimated_duration INTEGER
        )
    """)

    # Add new columns to existing databases if necessary
    cursor.execute("PRAGMA table_info(tasks)")
    existing_columns = [
        column[1] for column in cursor.fetchall()
    ]

    if "task_type" not in existing_columns:
        cursor.execute("""
            ALTER TABLE tasks
            ADD COLUMN task_type TEXT DEFAULT 'Cours'
        """)

    if "created_date" not in existing_columns:
        cursor.execute("""
            ALTER TABLE tasks
            ADD COLUMN created_date TEXT
        """)

    # Give existing tasks today's date as a migration fallback
    cursor.execute("""
        UPDATE tasks
        SET created_date = DATE('now')
        WHERE created_date IS NULL
    """)

    # 2. Activities: Study, Gym, Volleyball
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            activity_type TEXT NOT NULL
                CHECK(activity_type IN ('Study', 'Gym', 'Volleyball')),
            title TEXT NOT NULL,
            duration INTEGER NOT NULL,
            activity_date TEXT NOT NULL,
            notes TEXT
        )
    """)

    # 3. Personal habits
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 4. Daily habit tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS habit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0
                CHECK(completed IN (0, 1)),
            FOREIGN KEY (habit_id) REFERENCES habits(id),
            UNIQUE(habit_id, log_date)
        )
    """)

    connection.commit()
    connection.close()

    print("LifeOS database tables created successfully!")


if __name__ == "__main__":
    create_tables()

