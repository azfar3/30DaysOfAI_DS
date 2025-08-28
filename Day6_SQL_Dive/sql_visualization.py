# sql_visualization.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

conn = sqlite3.connect("google_play_store.db")

# Visualization 1: Top Categories by Number of Apps
query_viz1 = """
SELECT Category, COUNT(*) as num_apps 
FROM apps 
GROUP BY Category 
ORDER BY num_apps DESC 
LIMIT 10;
"""
viz1 = pd.read_sql_query(query_viz1, conn)

plt.subplot(2, 2, 1)
sns.barplot(data=viz1, x="num_apps", y="Category", palette="viridis")
plt.title("Top 10 Categories by Number of Apps")
plt.xlabel("Number of Apps")

# Visualization 2: Average Rating by Category
query_viz2 = """
SELECT Category, AVG(Rating) as avg_rating 
FROM apps 
GROUP BY Category 
HAVING COUNT(*) > 10 
ORDER BY avg_rating DESC 
LIMIT 10;
"""
viz2 = pd.read_sql_query(query_viz2, conn)

plt.subplot(2, 2, 2)
sns.barplot(data=viz2, x="avg_rating", y="Category", palette="magma")
plt.title("Top 10 Categories by Average Rating")
plt.xlabel("Average Rating")
plt.xlim(4, 5)

# Visualization 3: Free vs Paid Comparison
query_viz3 = """
SELECT Type, COUNT(*) as count, AVG(Rating) as avg_rating 
FROM apps 
GROUP BY Type;
"""
viz3 = pd.read_sql_query(query_viz3, conn)

plt.subplot(2, 2, 3)
sns.barplot(data=viz3, x="Type", y="avg_rating", palette="Set2")
plt.title("Average Rating: Free vs Paid Apps")
plt.ylabel("Average Rating")

# Visualization 4: Rating Distribution
query_viz4 = "SELECT Rating FROM apps WHERE Rating > 0;"
viz4 = pd.read_sql_query(query_viz4, conn)

plt.subplot(2, 2, 4)
sns.histplot(data=viz4, x="Rating", bins=20, kde=True)
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")

plt.tight_layout()
plt.savefig("sql_analysis_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

conn.close()
