import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import datetime
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn import Client
from libcomcat.search import search
from libcomcat.dataframes import get_summary_data_frame
import time
import glob
import os
import pytz
import csv
import re
import matplotlib.dates as mdates
import geopy.distance
from obspy.taup import TauPyModel
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt

##Instead use csv file, we will use obspy to get the event from Mt Rainier and a radius of 30 km

#def search_and_get_summary():
#    """Function to perform the event search and return a summary DataFrame."""
#    client = FDSN_Client("IRIS")
#    events = client.get_events(starttime=datetime(2023, 8, 25, 0, 0), endtime=datetime(2023, 8, 30, 0, 0),
#                               latitude=46.879967, longitude=-121.726906, maxradius=35/111.32)
#    event_df = get_summary_data_frame(events)
#    return event_df

#def get_summary_data_frame(events):
#    """Convert events to a summary DataFrame """
#    return pd.DataFrame({'EventID': [event.resource_id for event in events],
#                        'time': [event.origins[0].time for event in events]})

#event_df = event_df.sort_values(by=['time'],ascending=True)
#df_time = event_df['time']



# Paths

def find_files_with_dates(df_time, files_folder_path):
    """Find files in the specified directory that match the dates in the DataFrame."""
    original_dates = df_time.astype(str).tolist()
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')
    found_files = []

    directory_files = os.listdir(files_folder_path)
    
    for date_to_search in original_dates:
        match = date_pattern.match(date_to_search)
        if match:
            formatted_date = match.group(1).replace(" ", "_").replace(":", ".") + ".00"

            for file_name in directory_files:
                if formatted_date in file_name:
                    found_files.append(file_name)

    return found_files, original_dates


# Find the files corresponding to the dates in the DataFrame
#found_files, original_dates = find_files_with_dates(event_df['time'], files_folder_path)

## More fuctions
def adjust_seconds(currenttime_datetime):
    """Adjust the seconds of the datetime to match the file naming convention."""
    return currenttime_datetime.strftime('%Y-%m-%d_%H.%M.%S.%f')[:-3]  # Adjust as needed

def find_next_file(current_file_name):
    """Find the name of the next file."""
    folder_path = os.path.dirname(current_file_name)
    file_name = os.path.basename(current_file_name)
    # Correct extraction of date and time part
    file_time_str = '_'.join(file_name.split('_')[-3:-1])
    file_time = datetime.strptime(file_time_str, "%Y-%m-%d_%H.%M.%S")
    next_file_time = file_time + timedelta(minutes=1)
    next_file_name = file_name.replace(
        f"{file_time.year:04d}-{file_time.month:02d}-{file_time.day:02d}_{file_time.hour:02d}.{file_time.minute:02d}.{file_time.second:02d}",
        f"{next_file_time.year:04d}-{next_file_time.month:02d}-{next_file_time.day:02d}_{next_file_time.hour:02d}.{next_file_time.minute:02d}.{next_file_time.second:02d}"
    )
    next_file_path = os.path.join(folder_path, next_file_name)
    return next_file_path

def process_files_to_cut(found_files, original_dates, files_folder_path, output_folder_path, chan_min, chan_max, template_size):
    """Process the found files and save the cut data."""
    for file_name in found_files:
        data_file_path = os.path.join(files_folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        try:
            # Load the data and time using the load_file_data function
            data, timestamps = load_file_data(data_file_path, chan_min, chan_max)
            timestamps_utc = convert_timestamps_to_utc_np(timestamps)

            date_to_search = original_dates[found_files.index(file_name)]
            start_date = datetime.strptime(date_to_search, "%Y-%m-%d %H:%M:%S.%f") + timedelta(seconds=2)
            end_date = start_date + timedelta(seconds=template_size)

            # If the template needed is bigger that remain time on the file concatenate the next one!
            if (timestamps_utc[-1] < end_date):
                next_file_path = find_next_file(data_file_path)
                if os.path.exists(next_file_path):
                    concatenated_data, concatenated_timestamps = concatenate_files(data_file_path, next_file_path, chan_min, chan_max)
                    data = concatenated_data
                    timestamps = concatenated_timestamps
                    timestamps_utc = convert_timestamps_to_utc_np(timestamps)

            start_index = np.argmin(np.abs(np.array(timestamps_utc) - start_date))
            end_index = np.argmin(np.abs(np.array(timestamps_utc) - end_date))

            cut_data = data[start_index:end_index]
            cut_timestamps = timestamps[start_index:end_index]

            with h5py.File(output_file_path, 'w') as output_file:
                output_file.create_dataset('Acquisition/Raw[0]/RawData', data=cut_data)
                output_file.create_dataset('Acquisition/Raw[0]/RawDataTime', data=cut_timestamps)

        except Exception as e:
            print(f"Error procesando archivo {file_name}: {e}")

            
#Example of how to use process_file
# process_files(found_files, original_dates, files_folder_path, output_folder_path, chan_min, chan_max, template_size)



def convert_timestamps_to_utc_np(timestamps):
    """Convert timestamps to UTC format."""
    return [datetime.utcfromtimestamp(ts / 1e6) for ts in timestamps]

def load_file_data(data_file_path, chan_min = None, chan_max= None):
    """Load data from an HDF5 file."""
    with h5py.File(data_file_path, 'r') as data_file:
        if chan_min is not None and chan_max is not None:
            data = np.array(data_file['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max])
        else:
            data = np.array(data_file['Acquisition/Raw[0]/RawData'])
        timestamps = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])
    return data, timestamps

def concatenate_files(current_file_path, next_file_path, chan_min, chan_max):
    """Concatenate data from the current and the next file."""
    # Load data from the current file
    current_data, current_timestamps = load_file_data(current_file_path)
    current_data = current_data[:, chan_min:chan_max]

    # Load data from the next file
    next_data, next_timestamps = load_file_data(next_file_path)
    next_data = next_data[:, chan_min:chan_max]

    # Concatenate the data
    concatenated_data = np.concatenate((current_data, next_data), axis=0)
    concatenated_timestamps = np.concatenate((current_timestamps, next_timestamps), axis=0)

    return concatenated_data, concatenated_timestamps

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """Apply bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = filtfilt(b, a, data, axis=0)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Data length: {len(data)}. Required minimum length: {2 * max(len(a), len(b))}")
        return None
    return y

def find_indices_within_time_range(timestamps, start_time, end_time):
    """Find the indices of the timestamps within the specified range."""
    start_index = next(i for i, ts in enumerate(timestamps) if ts >= start_time)
    end_index = next(i for i, ts in enumerate(timestamps) if ts > end_time)
    return start_index, end_index

#This fuction check the shape of the templates, if not a need!

def check_saved_data_shapes(output_folder_path):
    """Check the shapes of the saved data in the specified directory."""
    for file_name in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, file_name)
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                raw_data_shape = f['Acquisition/Raw[0]/RawData'].shape
                raw_data_time_shape = f['Acquisition/Raw[0]/RawDataTime'].shape
                print(f"File: {file_name}")
                print(f"  RawData shape: {raw_data_shape}")
                print(f"  RawDataTime shape: {raw_data_time_shape}")
