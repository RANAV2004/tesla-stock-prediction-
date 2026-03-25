import pandas as pd

# Load data
df = pd.read_csv("../../data/tsla.csv")

# Check raw data
print("Before cleaning:")
print(df.head())

# Convert to numeric (this removes TSLA)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop invalid rows
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

# Final check
print("After cleaning:")
df.head()