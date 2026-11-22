
import pandas as pd

# Synthetic study-task data
data = [
    [30, "High", "Monday", 0.80, 1],
    [120, "Low", "Tuesday", 0.40, 0],
    [45, "Medium", "Wednesday", 0.75, 1],
    [90, "High", "Friday", 0.50, 0],
    [20, "Low", "Saturday", 0.90, 1],
    [60, "High", "Monday", 0.70, 1],
    [150, "Medium", "Tuesday", 0.30, 0],
    [40, "Low", "Wednesday", 0.60, 1],
    [100, "High", "Thursday", 0.45, 0],
    [25, "Medium", "Friday", 0.85, 1],
    [75, "High", "Saturday", 0.65, 1],
    [110, "Low", "Sunday", 0.35, 0],
    [35, "Medium", "Monday", 0.80, 1],
    [130, "High", "Tuesday", 0.25, 0],
    [50, "Low", "Thursday", 0.70, 1],
    [80, "Medium", "Friday", 0.55, 0],
    [15, "High", "Saturday", 0.95, 1],
    [95, "Low", "Sunday", 0.40, 0],
    [55, "High", "Wednesday", 0.75, 1],
    [140, "Medium", "Thursday", 0.30, 0],
]

columns = [
    "duration_minutes",
    "priority",
    "day_of_week",
    "previous_completion_rate",
    "completed"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset as CSV
df.to_csv("data/ml_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df["completed"].value_counts())
