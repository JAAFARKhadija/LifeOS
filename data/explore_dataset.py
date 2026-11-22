
import pandas as pd

# Load our dataset
df = pd.read_csv("data/ml_dataset.csv")

# 1. Display the first 5 rows
print("=== FIRST 5 ROWS ===")
print(df.head())

# 2. Dataset dimensions
print("\n=== DATASET SHAPE ===")
print(df.shape)

# 3. Column information and data types
print("\n=== DATASET INFO ===")
df.info()

# 4. Statistical summary
print("\n=== STATISTICAL SUMMARY ===")
print(df.describe())

# 5. Check missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 6. Count completed and incomplete tasks
print("\n=== TARGET DISTRIBUTION ===")
print(df["completed"].value_counts())

# 7. Completion rate
print("\n=== COMPLETION RATE ===")
print(df["completed"].mean())
