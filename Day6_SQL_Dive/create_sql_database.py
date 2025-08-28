# create_sql_database.py
import sqlite3
import pandas as pd
from pathlib import Path

# Load your cleaned data from Day 1
df = pd.read_csv("../Day1_DataWrangling/data/cleaned_googleplaystore.csv")

# Create a SQLite database
db_path = "google_play_store.db"
conn = sqlite3.connect(db_path)

# Save the DataFrame to SQL
df.to_sql("apps", conn, if_exists="replace", index=False)

# Verify the table was created
result = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:", result["name"].tolist())

# Check the schema
schema = pd.read_sql_query("PRAGMA table_info(apps);", conn)
print("\nTable schema:")
print(schema)

# Check sample data
sample = pd.read_sql_query("SELECT * FROM apps LIMIT 5;", conn)
print("\nSample data:")
print(sample)

conn.close()
print(f"\nDatabase created successfully: {db_path}")
