import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Adjust the pine green color here (darker shade)
pine_green_rgb = (1/255, 57/255, 41/255)  # Darker pine green color (R:1, G:57, B:41)

# Convert RGB to hex color code for matplotlib
pine_green_hex = '#%02x%02x%02x' % (int(pine_green_rgb[0]*255), int(pine_green_rgb[1]*255), int(pine_green_rgb[2]*255))

# Generate random data for adjective frequency (x-axis) and perceived emotion (y-axis)
np.random.seed(42)
adjective_frequency = np.random.randint(1, 100, 50)  # Random integers representing adjective frequency
perceived_emotion = adjective_frequency * np.random.uniform(0.5, 1.5) + np.random.normal(0, 10, 50)  # Simulating a relationship

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(adjective_frequency, perceived_emotion)

# Create the regression line
regression_line = slope * adjective_frequency + intercept

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(adjective_frequency, perceived_emotion, color='darkgrey', s=50)  # Set all points to dark grey
plt.scatter(adjective_frequency[25], perceived_emotion[25], color=pine_green_hex, s=75, label='Your Article')  # Highlighted point in pine green
plt.plot(adjective_frequency, regression_line, color=pine_green_hex, alpha=0.7, label=f'Regression Line\n$R^2 = {r_value**2:.2f}$')  # Semi-translucent pine green line

# Adding labels and title with Arial font
plt.xlabel('PCA1: Adjective Frequency', fontname='Arial', fontsize=14)
plt.ylabel('PCA2: GPT Perceived Emotion (Score)', fontname='Arial', fontsize=14)
plt.title('Linear Regression of Adjective Frequency vs. GPT Perceived Emotion', fontname='Arial', fontsize=16)
plt.legend(prop={'family': 'Arial', 'size': 12})
plt.grid(True)

# Show the plot
plt.show()
