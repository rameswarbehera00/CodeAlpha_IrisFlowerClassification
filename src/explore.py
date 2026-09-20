import os
import pandas as pd

# Path to the dataset
data_path = os.path.join("data", "Iris.csv")

# 1. Load CSV
df = pd.read_csv(data_path)

# 2. Inspect first 5 rows
print("--- FIRST 5 ROWS ---")
print(df.head())

# 3. Check data types and missing values
print("\n--- DATASET SUMMARY ---")
print(df.info())

# 4. Check for null values
print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())

# 5. Check class counts
print("\n--- SPECIES COUNT ---")
print(df["Species"].value_counts())

# 6. Five-number summary of numerical columns
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())