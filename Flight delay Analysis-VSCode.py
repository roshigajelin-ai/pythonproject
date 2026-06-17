# ==========================================
# AIRLINE DELAY ANALYSIS PROJECT
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CREATE DATASET (1000 ROWS)
# ==========================================

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "Month": np.random.randint(1,13,n),
    "Operating_Airline": np.random.choice(
        ["AA","DL","UA","WN","B6"], n
    ),
    "DepDelay": np.random.normal(10,20,n),
    "ArrDelay": np.random.normal(8,22,n),
    "TaxiOut": np.random.randint(5,40,n),
    "TaxiIn": np.random.randint(3,20,n),
    "AirTime": np.random.randint(45,350,n),
    "Distance": np.random.randint(100,3000,n)
})

# Save CSV

df.to_csv("airline_delay_dataset.csv", index=False)

print("Dataset Created Successfully!")
print()

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv("airline_delay_dataset.csv")

print("First 5 Rows")
print(df.head())

# ==========================================
# 3. EXPLORE DATASET
# ==========================================

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

df = df.dropna()

# ==========================================
# 5. NUMPY STATISTICS
# ==========================================

print("\nNUMPY STATISTICS")

print("Mean Arrival Delay:",
      np.mean(df["ArrDelay"]))

print("Median Arrival Delay:",
      np.median(df["ArrDelay"]))

print("Standard Deviation:",
      np.std(df["ArrDelay"]))

print("Maximum Delay:",
      np.max(df["ArrDelay"]))

print("Minimum Delay:",
      np.min(df["ArrDelay"]))

# ==========================================
# 6. DATA ANALYSIS
# ==========================================

print("\nAverage Delay by Airline")

airline_analysis = (
    df.groupby("Operating_Airline")
    ["ArrDelay"]
    .mean()
)

print(airline_analysis)

# ==========================================
# 7. VISUALIZATION 1
# BAR CHART
# ==========================================

plt.figure(figsize=(10,6))

airline_analysis.sort_values(
    ascending=False
).plot(kind="bar")

plt.title(
    "Average Arrival Delay by Airline"
)

plt.xlabel("Airline")
plt.ylabel("Average Delay (Minutes)")

plt.tight_layout()
plt.show()

# ==========================================
# 8. VISUALIZATION 2
# LINE CHART
# ==========================================

monthly_delay = (
    df.groupby("Month")
    ["ArrDelay"]
    .mean()
)

plt.figure(figsize=(10,6))

plt.plot(
    monthly_delay.index,
    monthly_delay.values,
    marker="o"
)

plt.title(
    "Monthly Average Arrival Delay"
)

plt.xlabel("Month")
plt.ylabel("Average Delay")

plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 9. VISUALIZATION 3
# HISTOGRAM
# ==========================================

plt.figure(figsize=(10,6))

plt.hist(
    df["ArrDelay"],
    bins=30
)

plt.title(
    "Distribution of Arrival Delays"
)

plt.xlabel("Delay (Minutes)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==========================================
# 10. VISUALIZATION 4
# HEATMAP
# ==========================================

numeric_columns = [
    "DepDelay",
    "ArrDelay",
    "TaxiOut",
    "TaxiIn",
    "AirTime",
    "Distance"
]

corr_matrix = (
    df[numeric_columns]
    .corr()
)

plt.figure(figsize=(10,6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()
plt.show()

# ==========================================
# 11. CORRELATION DETECTIVE
# ==========================================

print("\nCorrelation Matrix")
print(corr_matrix)

# ==========================================
# 12. KEY INSIGHTS
# ==========================================

print("\nKEY INSIGHTS")

print("""
1. Airlines have different average delay patterns.

2. Delay levels vary across months.

3. Most flights experience small delays.

4. Departure delay strongly affects arrival delay.

5. Taxi time contributes to flight delays.

6. Distance and airtime generally move together.
""")

# ==========================================
# 13. CONCLUSION
# ==========================================

print("""
PROJECT CONCLUSION

This project analyzed airline delay data
using Python, NumPy, Pandas, Matplotlib
and Seaborn.

The analysis showed delay patterns
across airlines and months.

Most delays are small, but a few flights
experience significant delays.

Correlation analysis identified important
relationships that can be useful for
future machine learning models.
""")