# sql_queries.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the database
conn = sqlite3.connect("google_play_store.db")

# Problem 1: Basic Aggregation
print("=== PROBLEM 1: Basic Aggregation ===")
query_1 = """
SELECT 
    Category,
    COUNT(*) as num_apps,
    AVG(Rating) as avg_rating,
    AVG(Reviews) as avg_reviews
FROM apps
GROUP BY Category
ORDER BY num_apps DESC
LIMIT 10;
"""
result_1 = pd.read_sql_query(query_1, conn)
print(result_1)

# Problem 2: Free vs Paid Analysis
print("\n=== PROBLEM 2: Free vs Paid Analysis ===")
query_2 = """
SELECT 
    Type,
    COUNT(*) as count,
    AVG(Rating) as avg_rating,
    AVG(Reviews) as avg_reviews,
    SUM(Installs) as total_installs
FROM apps
GROUP BY Type;
"""
result_2 = pd.read_sql_query(query_2, conn)
print(result_2)

# Problem 3: Window Functions - Top Apps per Category
print("\n=== PROBLEM 3: Top Apps per Category ===")
query_3 = """
WITH ranked_apps AS (
    SELECT 
        App,
        Category,
        Rating,
        Reviews,
        Installs,
        ROW_NUMBER() OVER (PARTITION BY Category ORDER BY Reviews DESC) as rank
    FROM apps
    WHERE Reviews > 0
)
SELECT * FROM ranked_apps WHERE rank <= 3;
"""
result_3 = pd.read_sql_query(query_3, conn)
print(result_3.head(12))  # Show top 3 from first few categories

# Problem 4: Correlation Analysis using SQL
print("\n=== PROBLEM 4: Correlation Analysis ===")
query_4 = """
SELECT 
    AVG(Rating) as avg_rating,
    AVG(Reviews) as avg_reviews,
    AVG(Installs) as avg_installs,
    AVG(Price) as avg_price,
    AVG(Size) as avg_size
FROM apps;
"""
result_4 = pd.read_sql_query(query_4, conn)
print(result_4)

# Problem 5: Complex Conditional Analysis
print("\n=== PROBLEM 5: Conditional Analysis ===")
query_5 = """
SELECT 
    CASE 
        WHEN Rating >= 4.5 THEN 'Excellent (4.5-5.0)'
        WHEN Rating >= 4.0 THEN 'Good (4.0-4.4)'
        WHEN Rating >= 3.5 THEN 'Average (3.5-3.9)'
        ELSE 'Poor (<3.5)'
    END as rating_category,
    COUNT(*) as num_apps,
    AVG(Reviews) as avg_reviews,
    AVG(Installs) as avg_installs
FROM apps
GROUP BY rating_category
ORDER BY num_apps DESC;
"""
result_5 = pd.read_sql_query(query_5, conn)
print(result_5)

# Problem 6: Date Analysis (if you have date fields)
print("\n=== PROBLEM 6: Date Analysis ===")
query_6 = """
SELECT 
    Category,
    AVG(`Last Updated`) as avg_days_since_update,
    COUNT(*) as num_apps
FROM apps
GROUP BY Category
HAVING num_apps > 10
ORDER BY avg_days_since_update DESC
LIMIT 10;
"""
result_6 = pd.read_sql_query(query_6, conn)
print(result_6)

# Close connection
conn.close()
