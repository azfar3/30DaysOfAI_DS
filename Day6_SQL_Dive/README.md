# Day 6: SQL Deep Dive - Analyzing App Data

## Objective
Practice advanced SQL techniques by analyzing the Google Play Store dataset.

## What Was Covered
- **Database Creation**: Converted pandas DataFrame to SQLite database
- **Basic SQL**: Aggregations, GROUP BY, ORDER BY
- **Advanced SQL**: Window functions, CTEs, conditional logic
- **Complex Queries**: Multiple joins, subqueries, advanced filtering
- **Data Visualization**: Creating plots from SQL results

## Key SQL Concepts Practiced
1. **Aggregate Functions**: COUNT, SUM, AVG, MIN, MAX
2. **Window Functions**: ROW_NUMBER, RANK, running totals
3. **CTEs (Common Table Expressions)**: WITH clauses
4. **CASE Statements**: Conditional logic in SQL
5. **Advanced Joins**: Complex relationship analysis

## Files Created
- `create_sql_database.py`: Creates SQLite database from cleaned google playstore data
- `sql_queries.py`: Basic to intermediate SQL practice problems
- `advanced_sql_challenges.py`: Complex SQL challenges
- `sql_visualization.py`: Visualizations from SQL results
- `google_play_store.db`: SQLite database file

## Sample Insights Discovered
- Top categories by number of apps
- Average ratings by category
- Free vs Paid app analysis
- Correlation between reviews and ratings
- Hidden gem apps with high ratings but low reviews

## How to Run
```bash
# Run the scripts in order
python create_sql_database.py
python sql_queries.py
python advanced_sql_challenges.py
python sql_visualization.py
```

## Skills Developed
- SQL query optimization
- Data analysis using SQL
- Database management
- Results interpretation
- Data visualization from SQL results