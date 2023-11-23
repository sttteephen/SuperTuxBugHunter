import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def multiple_histograms():
    # Paths to the specific files
    file_paths = ['/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_1dfb376f-b824-4d5e-868c-d5820bb2e3aa.txt',
                '/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_6dff6bf0-ad84-43a7-bdde-1dcd65297b42.txt',
                '/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_44a1ef3e-b66e-465f-937f-c69bb807b683.txt',
                '/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_369daa66-2086-42ab-963f-8f1a96309d52.txt',
                '/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_1805345c-1d4b-46dd-936b-dc40afba5ea7.txt',
                '/home/stephen/pystk_wrapper/SKP_FRAME_4_ENV_f99bc3a0-10a0-492c-ac58-13cc61287054.txt']

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
        #plt.axvline(x=0.00148574816, color='red', linestyle='--', linewidth=2)  # Draw the vertical line
        plt.title(f'File {i} Histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def single_histogram():
    # Replace this with the path to your CSV file
    file_path = '/home/stephen/pystk_wrapper/same_action_fps7d869b53-bc18-4f21-93ed-268ab1730c61.csv'

    # Read the CSV file
    # Assuming the CSV file has a header and the data is in the first column
    # If the data is in a different column, adjust the column index accordingly
    df = pd.read_csv(file_path)

    # Flatten the DataFrame to get a single list of all values
    all_values = df.values.flatten()

    # Plotting the histogram
    plt.figure(figsize=(28, 12))
    plt.hist(all_values, bins=200, color='blue', edgecolor='black', alpha=0.7)
    #plt.axvline(x=0.00148574816, color='red', linestyle='--', linewidth=2)  # Draw the vertical line if needed
    plt.title('Histogram of Aggregated Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    input()

if __name__ == "__main__":
    single_histogram()