import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
import datetime
import pandas as pd
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import time
import glob
import os
import pytz
from tqdm import tqdm
import csv
import matplotlib.dates as mdates
import csv
import re
from templatematching import *

#loading data to run

def get_file_list(base_path,start_date,end_date):
    file_list = []
    actual_date = start_date
    
    while actual_date <= end_date:
        #new_decimator is because is downsample data, usually is just decimator
        search_pattern = f"{base_path}/new_decimator_{actual_date}*"
        file_list.extend(glob.glob(search_pattern))
        actual_date = increase_date(actual_date)
        file_list.sort()
    return file_list
    
# defining the other fuction to increment the date
def increase_date(date_str):
    from datetime import datetime,timedelta
    date_format = "%Y-%m-%d_%H"
    date = datetime.strptime(date_str,date_format)
    next_date = date + timedelta(hours=1)
    return next_date.strftime(date_format)

# Buiding outputfiles and correlations for each template on the list

# Fuctions
def loading_data(file, tem, chan_min, chan_max):
    '''
    Load data from an HDF5 file and a template file, and return the template, 
    raw data filtered by channel, and timestamps.

    Args:
        file (str): Path to the raw data file.
        tem (str): Path to the template file.
        chan_min (int): Index of the minimum channel to consider.
        chan_max (int): Index of the maximum channel to consider (exclusive).

    Returns:
        tuple: A tuple containing:
            - template (np.ndarray): Template data.
            - raw_data (np.ndarray): Raw data filtered by channel.
            - timestamps (np.ndarray): Timestamps of the raw data.

    Raises:
        RuntimeError: If there is an error while loading the data.
    
    '''
    try:
        with h5py.File(file, "r") as f, h5py.File(tem, "r") as d:
            template = np.array(d['Acquisition/Raw[0]/RawData'][:, 0:-1])
            raw_data = np.array(f['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max-1])
            timestamps = np.array(f['Acquisition/Raw[0]/RawDataTime'])
        return template, raw_data, timestamps
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def filter_data(raw_data, b, a):
    """
    Filtering the raw_data using filfilt
    
    b,a comes from butter 
    """
    try:
        data_filt = filtfilt(b, a, raw_data, axis=0)
        return data_filt
    except Exception as e:
        raise RuntimeError(f"Error filtering data: {e}")

def compute_correlations(template, data_filt, samples_per_file, channel_number):
    """
    This fuction calculate the correlation between the template with data filt
    the correlation is the size of the template.
    
    corr : use the fuction window_and_correlate that is templatematching.py
    
    corrs2: it is the reorganizzation the correlations per channel 
    
    corrs3: it is average of the correlation value per channel
    all of that it is calculate per one file
    
    """
    
    try:
        corrs = window_and_correlate(template, data_filt)
        corrs2 = corrs.reshape((int(samples_per_file), channel_number-1))
        corrs3 = np.sum(corrs2, axis=1) / channel_number
        return corrs3
    except Exception as e:
        raise RuntimeError(f"Error computing correlations: {e}")

def process_files(file_list, template_list, chan_min, chan_max, channel_number, samples_per_file, b, a, full_path):
    """
    This fuction calculate the correlation between the template with data filt
    the correlation is the size of the template.
    
    The loop is opening 1 file from file_list and correlating them with all templates saved on template list
    Next it is doing the same but the next file in file_list
    
    
    """
    
    start_time = time.perf_counter()
    
    for i, file in tqdm(enumerate(file_list)):
        for j, tem in tqdm(enumerate(template_list)):
            try:
                # Load data
                template, raw_data, timestamps = loading_data(file, tem, chan_min, chan_max)
                
                # Filter
                data_filt = filter_data(raw_data, b, a)
                
                # Compute correlations
                corrs3 = compute_correlations(template, data_filt, samples_per_file, channel_number)
                
                # output folder for correlations, based on the name of the template
                folder_name_parts = os.path.splitext(os.path.basename(tem))[0].split('_')[2:4]
                folder_name = '_'.join(folder_name_parts)
                folder_output = os.path.join(full_path, folder_name)
                
                # create it if does not exist
                if not os.path.exists(folder_output):
                    os.mkdir(folder_output)
                
                # Saved correlations values
                outfile_name = os.path.join(folder_output, f'corrs_{i}_.npy')
                np.save(outfile_name, corrs3)
                print(f"Saved: {outfile_name}")
            except Exception as e:
                print(f"Error processing file {file} with template {tem}: {e}. Skipping this file and moving to the next template.")
                continue

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The code took {execution_time} seconds.")
    
# ANALYSIS OF DATA

# Making .h5 with the date from file_list


def create_timestamps_h5(file_list, output_dir):
    """
    Creates an HDF5 file with timestamps extracted from the given list of files.

    Args:
        file_list (list): List of file paths containing the timestamps.
        output_dir (str): Directory where the HDF5 file will be saved.

    Returns:
        str: File path of the created HDF5 file.
    """
    # Extract date from the first and last file names
    first_file_name = os.path.basename(file_list[0])
    file_name_parts = first_file_name.split('_')
    date_time_part = '_'.join(file_name_parts[2:4])  # '2023-08-27_11.00.00_UTC'


    last_file_name = os.path.basename(file_list[-1])
    last_file_name_parts = last_file_name.split('_')
    last_date_time_part = '_'.join(last_file_name_parts[2:4])  # '2023-08-27_11.00.00_UTC'
    
    # Create the output filename with the desired format
    output_file_name = f"timestamps_{date_time_part}_{last_date_time_part}.h5"
    output_file_h5 = os.path.join(output_dir, output_file_name)

    ## Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the HDF5 file and save the timestamps

    with h5py.File(output_file_h5, 'w') as f:
    # Create a dataset to store the timestamps
        timestamps_dataset = f.create_dataset('timestamps', (0,), dtype='i8', maxshape=(None,))

    # Iterate over the files and add the timestamps to the dataset
        for file_path in tqdm(file_list, desc="Processing files"):
            with h5py.File(file_path, 'r') as f_read:
            # Get timestamps from the read file
                timestamps = np.array(f_read['Acquisition/Raw[0]/RawDataTime'])
                num_timestamps = len(timestamps)

            # Extend the dataset to add the new timestamps
                timestamps_dataset.resize((timestamps_dataset.shape[0] + num_timestamps,))
                timestamps_dataset[-num_timestamps:] = timestamps

# Print completion message
    print(f"Timestamps have been saved in the file {output_file_h5}.")
    return output_file_h5

def convert_timestamps_to_utc(input_file_h5,pt_timezone_str='America/Los_Angeles',utc_timezone_str='UTC'):
    with h5py.File(input_file_h5, 'r') as f:
        timestamps_pt = np.array(f['timestamps'])

    # # Define the Pacific Timezone
    pt_timezone = pytz.timezone(pt_timezone_str)

    # Convert timestamps from microseconds to seconds
    timestamps_seconds = timestamps_pt / 1e6

    # Convert timestamps to datetime objects in Pacific Time
    datetime_objects_pt = [datetime.datetime.fromtimestamp(ts, pt_timezone) for ts in timestamps_seconds]

    # Define the UTC Timezone
    utc_timezone = pytz.timezone('UTC')

    # Convert datetime objects to UTC
    time_utc = [dt_pt.astimezone(utc_timezone) for dt_pt in datetime_objects_pt]
    
    return time_utc


# Parker approach
from scipy.stats import norm

def mad_func_li(arr):
    """ Median Absolute Deviation: Using the formulation in Li and Zhan 2018
    Pushing the limit of earthquake detection with DAS and template matching
    """
    # arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    mean = np.mean(arr)
    med = np.median(arr)
    return np.median(np.abs(mean - med))
def mad_func_shelly(arr):
    """ Median Absolute Deviation: Using the formulation in Li and Zhan 2018
    Pushing the limit of earthquake detection with DAS and template matching
    """
    # arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    # mean = np.mean(arr)
    med = np.median(arr)
    return np.median(np.abs(arr - med))
def detections(cc_arr, mode='shelly'):
    """ using detection significance from Li and Zhan 2018
    det = (peak - med) / mad
    """
    med = np.median(cc_arr)
    if mode == 'shelly':
        mad = mad_func_shelly(cc_arr)
    elif mode == 'li':
        mad = mad_func_li(cc_arr)
    det = (cc_arr - med) / mad
    return det,mad


# full_path is going to give you the CC data set for this run, also calcultin MAD so we can define the thresold
def process_folders(full_path, fs):
    try:
        # Get a list of all folders in the base directory
        folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]
    except Exception as e:
        print(f"Error listing folders in {full_path}: {e}")
        return

    # Initialize a list to store concatenated data for each folder
    concatenated_data_per_folder = []

    # Iterate over the folders and load the .npy files
    for folder in folders:
        folder_path = os.path.join(full_path, folder)
        try:
            npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
            concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            continue

    # Calculate MAD for each folder and define thresholds
    mads_per_folder = {}
    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        try:
            det_shelly, mad = detections(folder_data, mode='shelly')
            thresh = np.round(np.abs(norm.ppf(1 / fs / (60 * 60))))
            thresh2 = np.round(average_mad*12, decimals=3)
            mads_per_folder[folder] = np.round(mad,decimals=3)
            #print(f"MAD for folder {folder}: {mad}")
            print(f"Threshold for folder {folder}: {thresh}")
            print(f"Threshold2 for folder {folder}: {thresh2}")
            #det_ind = np.where(np.abs(det_shelly) > thresh)[0]
        except Exception as e:
            print(f"Error calculating MAD for folder {folder}: {e}")

    # Calculate and display the average MAD
    try:
        #average_mad = np.round(np.mean(list(mads_per_folder.values())),decimals=3)
        print(f"Average MAD: {average_mad}")
    except Exception as e:
        print(f"Error calculating average MAD: {e}")
#det_shelly,mad = detections(correlations_all,mode='shelly')
#thresh = np.round(np.abs(norm.ppf(1/fs/(15*60))))
# det_li = detections(correlations_all,mode='li')
# det_hil = np.abs(hilbert(det_shelly))
# det_ind_hil = np.where(det_hil > thresh)[0]
#det_ind = np.where(np.abs(det_shelly) > thresh)[0]
#print(f'Initial Num detections: {len(det_ind)}')
# print(f'Initial Num Hilbert detections: {len(det_ind_hil)}')








