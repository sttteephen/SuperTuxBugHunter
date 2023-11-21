import numpy as np
import matplotlib.pyplot as plt

# Paths to the specific files
file_paths = [
              '/home/stephen/SuperTuxBugHunter/reline_training_fps/TEST_FOURENV1700504459a2f02c25-289c-4bed-bc03-b56d5f7456be.txt',
              '/home/stephen/SuperTuxBugHunter/reline_training_fps/TEST_FOURENV170050423571d29c0b-0b90-443f-b4ce-30e7eac4676d.txt']

# Initialize a list to hold the data from each file
data_lists = []

# Read data from each file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file]
        data_lists.append(numbers)

# Create a subplot for each file's data
plt.figure(figsize=(28, 12))
for i, data in enumerate(data_lists, 1):
    plt.subplot(2, 4, i)  # 1 row, 4 columns, i-th subplot
    plt.hist(data, bins=120, color='blue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.00148574816, color='red', linestyle='--', linewidth=2)  # Draw the vertical line
    plt.title(f'File {i} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
