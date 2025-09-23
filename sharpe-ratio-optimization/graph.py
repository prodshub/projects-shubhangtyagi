import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "vanguard_implied_volatility.csv"  # Adjust if necessary
df = pd.read_csv(file_path)

# Convert "Implied Volatility" column to numeric (handling NaNs)
df["Implied Volatility"] = pd.to_numeric(df["Implied Volatility"], errors='coerce')

# Filter out rows where Implied Volatility is NaN or Market Price is zero
df = df[df["Market Price"] > 0]

# Sort by Strike Price for better visualization
df = df.sort_values(by="Strike Price")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df["Strike Price"], df["Implied Volatility"], marker='o', linestyle='-', color='b', label="Implied Volatility")

# Formatting
plt.xlabel("Strike Price ($)")
plt.ylabel("Implied Volatility")
plt.title(f"Implied Volatility vs. Strike Price ({df['Symbol'].iloc[0]})")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
