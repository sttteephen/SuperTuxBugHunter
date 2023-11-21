import os
import numpy as np

directory_path = 'reline_training_fps'  # Replace with your directory path

# Initialize an empty list to hold all numbers from all files
all_numbers = []

# Iterate over each file in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Ensure the file is a text file
    if os.path.isfile(file_path) and file_path.endswith('.txt')  and ('TEST_FOURENV1700504459a2f02c25-289c-4bed-bc03-b56d5f7456be.txt' in file_path or 'FPS_INFO_BASELINE.txt' in file_path):
        with open(file_path, 'r') as file:
            # Extend the list with numbers from this file
            all_numbers.extend([float(line.strip()) for line in file])

# Calculate mean, standard deviation, and IQR for the aggregated data
mean_value = np.mean(all_numbers)
std_dev = np.std(all_numbers)
Q1 = np.percentile(all_numbers, 25)
Q3 = np.percentile(all_numbers, 75)
IQR = Q3 - Q1

print(len(all_numbers))
above_thresh = [x for x in all_numbers if x > 0.00148574816]
print(len(above_thresh))
print(len(above_thresh) / len(all_numbers))
print("Aggregated Data Calculations:")
print("Mean:", mean_value)
print("Standard Deviation:", std_dev)
print('Threshold:', mean_value + (5*std_dev))
print("Interquartile Range:", IQR)
