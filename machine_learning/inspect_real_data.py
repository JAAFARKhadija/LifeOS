
from database.db import get_connection

connection = get_connection()
cursor = connection.cursor()

# 1. Check the task table structure
print("=== TASK TABLE STRUCTURE ===")

cursor.execute("PRAGMA table_info(tasks)")
for column in cursor.fetchall():
    print(column)

# 2. Count all real tasks
print("\n=== TOTAL TASKS ===")

cursor.execute("SELECT COUNT(*) FROM tasks")
print(cursor.fetchone()[0])

# 3. Count tasks by status
print("\n=== TASKS BY STATUS ===")

cursor.execute("""
    SELECT status, COUNT(*)
    FROM tasks
    GROUP BY status
""")

for row in cursor.fetchall():
    print(row)

# 4. Check how many tasks have missing values
print("\n=== MISSING VALUES ===")

cursor.execute("""
    SELECT
        SUM(CASE WHEN priority IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN due_date IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN estimated_duration IS NULL THEN 1 ELSE 0 END)
    FROM tasks
""")

print(cursor.fetchone())

connection.close()
