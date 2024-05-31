import h5py
from scipy.signal import butter, filtfilt, decimate
import numpy as np
import time
import glob
import os
from datetime import datetime


#create a patch when the copy is interrupted

def get_last_date(output_folder):
    processed_files = glob.glob(os.path.join(output_folder, 'new_decimator_*.h5'))
    if processed_files:
        try:
            # Extract the date from the last processed file
            last_file = processed_files[-1]
            date_str = last_file.split('_')[-3]  # Adjust index based on the file name format
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj
        except ValueError:
            print(f"Error: Unable to extract a valid date from the file {last_file}.")
    return None

file_list = glob.glob("/auto/petasaur-wd15/rainier-10-14-2023-drive2/rainier/*")
file_list.sort()

# Set frequencies
original_frequency = 200  # Hz
desire_frequency = 20    # Hz
chan_min = 0
chan_max = -1

# Butter function
orden_filter = 4
cutoff_frequency = desire_frequency / 2.0  # Frequency in Hz

# Define filter
b, a = butter(orden_filter, cutoff_frequency / (original_frequency / 2.0), btype='low')

# Iteration for file_list
output_folder = "/data/data4/veronica-scratch-rainier/drive2_ds"

# Find the last date
last_date = get_last_date(output_folder)

total_start_time = time.perf_counter()

# Continue from the last date recovered
for file_name in file_list:
    file_date_str = file_name.split('_')[-3]  # Adjust index based on the file name format
    file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

    if last_date is None or file_date > last_date:
        # Load HDF5 files
        with h5py.File(file_name, 'r') as file:
            # Obtain the dataset
            original_data = file['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max]
            original_time = np.array(file['Acquisition/Raw[0]/RawDataTime'])
            attrs = dict(file['Acquisition'].attrs)  # Copy attributes

        # Calculate decimation factor
        factor_decimation = original_frequency // desire_frequency

        # Apply the filter
        data_filtered = filtfilt(b, a, original_data, axis=0)

        # Decimate the result
        downsampled_data = decimate(data_filtered, factor_decimation, axis=0)

        # Decimate time as well
        downsampled_time = original_time[:original_time.shape[0]:factor_decimation]

        # New file name
        output_file_name = f"new_{file_name.split('/')[-1]}"

        # Save the data
        with h5py.File(os.path.join(output_folder, output_file_name), 'w') as g:
            g.create_dataset('Acquisition/Raw[0]/RawData', data=downsampled_data)
            g.create_dataset('Acquisition/Raw[0]/RawDataTime', data=downsampled_time, dtype='f8')
            g['Acquisition'].attrs.update(attrs)

total_end_time = time.perf_counter()

total_elapsed_time = total_end_time - total_start_time
print(f"Total time: {total_elapsed_time} seconds")