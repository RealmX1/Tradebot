import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate random data
np.random.seed(42)
num_points = 100

time = np.arange(num_points)
random_data1 = np.random.normal(5, 1, num_points)
random_data2 = np.random.normal(100, 10, num_points)

# Calculate Exponential Moving Average (EMA)
alpha = 0.1  # Smoothing factor
data1_ema = pd.Series(random_data1).ewm(alpha=alpha).mean()
data2_ema = pd.Series(random_data2).ewm(alpha=alpha).mean()

# Create the main figure and subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})

# 1. First subplot - Horizontal line
axes[0].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, num_points)
axes[0].set_title("Horizontal Line")
axes[0].set_yticks([])  # Hide y-axis ticks

# 2. Second subplot - Dual y-axis with 4 data series
ax1 = axes[1]  # Left y-axis
ax2 = ax1.twinx()  # Right y-axis

# Plot data on both axes
ax1.plot(time, random_data1, label="Random Data 1 (mean=5)", color='tab:blue')
ax1.plot(time, data1_ema, label="EMA (Random Data 1)", color='tab:cyan', linestyle='--')
ax2.plot(time, random_data2, label="Random Data 2", color='tab:orange')
ax2.plot(time, data2_ema, label="EMA (Random Data 2)", color='tab:red', linestyle='--')

# Calculate the range for both datasets
data1_range = random_data1.max() - random_data1.min()
data2_range = random_data2.max() - random_data2.min()

# Set y-axis limits to align the first points
data1_start = random_data1[0]
data2_start = random_data2[0]

# Calculate limits that maintain the same relative scale but align starting points
data1_min = data1_start - data1_range * 0.4
data1_max = data1_start + data1_range * 0.6
data2_min = data2_start - data2_range * 0.4
data2_max = data2_start + data2_range * 0.6

ax1.set_ylim(data1_min, data1_max)
ax2.set_ylim(data2_min, data2_max)

# Set axis labels and titles
ax1.set_xlabel("Time")
ax1.set_ylabel("Random Data 1 (Left Y-Axis)", color='tab:blue')
ax2.set_ylabel("Random Data 2 (Right Y-Axis)", color='tab:orange')

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# Set title for second subplot
axes[1].set_title("Dual Y-Axis Plot with EMA (Aligned Starting Values)")

# Adjust layout and show plot
plt.tight_layout()
plt.show()
