# advanced_sql_challenges.py
import sqlite3
import pandas as pd

conn = sqlite3.connect("google_play_store.db")

# Challenge 1: Find apps with high ratings but low reviews (hidden gems)
print("=== CHALLENGE 1: Hidden Gems ===")
challenge_1 = """
SELECT 
    App,
    Category,
    Rating,
    Reviews,
    Installs
FROM apps
WHERE Rating >= 4.5 AND Reviews < 1000
ORDER BY Rating DESC, Reviews
LIMIT 10;
"""
result_c1 = pd.read_sql_query(challenge_1, conn)
print(result_c1)

# Challenge 2: Price analysis by category
print("\n=== CHALLENGE 2: Price Analysis ===")
challenge_2 = """
SELECT 
    Category,
    COUNT(*) as total_apps,
    SUM(CASE WHEN Price > 0 THEN 1 ELSE 0 END) as paid_apps,
    ROUND(100.0 * SUM(CASE WHEN Price > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as percent_paid,
    AVG(CASE WHEN Price > 0 THEN Price ELSE NULL END) as avg_price
FROM apps
GROUP BY Category
HAVING total_apps > 10
ORDER BY percent_paid DESC
LIMIT 10;
"""
result_c2 = pd.read_sql_query(challenge_2, conn)
print(result_c2)

# Challenge 3: Popularity-Rating correlation
print("\n=== CHALLENGE 3: Popularity vs Rating ===")
challenge_3 = """
WITH popularity_buckets AS (
    SELECT 
        App,
        Category,
        Rating,
        Reviews,
        CASE 
            WHEN Reviews > 1000000 THEN 'Very Popular'
            WHEN Reviews > 100000 THEN 'Popular'
            WHEN Reviews > 10000 THEN 'Moderate'
            ELSE 'Niche'
        END as popularity
    FROM apps
    WHERE Reviews > 0
)
SELECT 
    popularity,
    COUNT(*) as num_apps,
    AVG(Rating) as avg_rating,
    MIN(Rating) as min_rating,
    MAX(Rating) as max_rating
FROM popularity_buckets
GROUP BY popularity
ORDER BY 
    CASE popularity
        WHEN 'Very Popular' THEN 1
        WHEN 'Popular' THEN 2
        WHEN 'Moderate' THEN 3
        WHEN 'Niche' THEN 4
    END;
"""
result_c3 = pd.read_sql_query(challenge_3, conn)
print(result_c3)

# Challenge 4: Advanced window function - Running total
print("\n=== CHALLENGE 4: Running Total ===")
challenge_4 = """
SELECT 
    Category,
    App,
    Reviews,
    SUM(Reviews) OVER (PARTITION BY Category ORDER BY Reviews DESC) as running_total,
    ROUND(100.0 * Reviews / SUM(Reviews) OVER (PARTITION BY Category), 2) as percent_of_category
FROM apps
WHERE Category = 'GAME'  -- Change to any category you want to analyze
ORDER BY Reviews DESC
LIMIT 10;
"""
result_c4 = pd.read_sql_query(challenge_4, conn)
print(result_c4)

conn.close()
