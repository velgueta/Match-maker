import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np
from datetime import datetime, timedelta
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
from scipy.stats import norm
from templatematching import *
from getanalisisfiles import *
from template_maker2 import *



#loading data to run, write here the path for your data.

base_path = "/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds"
#base_path = '/1-fnp/petasaur/p-wd15/rainier-10-14-2023-drive1/'
output_plot_directory = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/plots-CC_15sec-tem_2023-08-27_10.10-2023-08-27_10.30' # change it for any iteration
start_date_process = "2023-08-27_10.00" # choose your data window
end_date_process = "2023-08-27_10.59"
# using the fuction get_file_list 


file_list = get_file_list(base_path,start_date_process,end_date_process)

#  verify # of elements of file_list
if file_list:
    print(f"File list contains {len(file_list)} files.")
else:
    print("File list is empty.")

## templates
events = search(starttime = datetime(2023, 8, 26, 0, 0), 
                endtime = datetime(2023,8,31,0,0),
                #endtime   = datetime.datetime.now(),
                latitude=46.879967,
                longitude=-121.726906,
                maxradius= 35/111.32) 
                #maxradius= 20) 
event_df = get_summary_data_frame(events)
print("Returned %s events" % len(events))

# Calculate the templates

# Sort the DataFrame and extract the 'time' column
event_df = event_df.sort_values(by=['time'], ascending=True)
#df_time = event_df['time']

##  Find the files corresponding to the dates with their location in the DataFrame
#matches_files, original_dates = find_files_with_dates(event_df, base_path)
matched_files_with_locations = find_files_with_dates(event_df, base_path)

found_files = [item['matched_file'] for item in matched_files_with_locations]
original_dates = [item['original_date'] for item in matched_files_with_locations]


# Parameters

chan_min = 0
chan_max = 3000
channel_number = (chan_max -chan_min)
template_size = 15  # In seconds # with 8 sec did not work!
fs = 20  # Sampling frequency, this should be directly from atts of the files in drive1_ds
samples_per_file = 60*fs # should be integer number
#print(samples_per_file)

# Paths

files_folder_path = '/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds' # where to find the raw data
#files_folder_path = '/1-fnp/petasaur/p-wd15/rainier-10-14-2023-drive1/' #original data with no downsampling                                                   
#output_folder_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/template-two-seconds/' # where to save the templates
output_folder_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/template-15second-test/' # where to save the templates
                                                   
#test2 = 3 sec., test=6 test3 = ?

# Fuction to generate raw templates, template_make2.py

process_files_to_cut(found_files, original_dates, base_path , output_folder_path, chan_min, chan_max, template_size) #this duction just need to be run once!


low_cut = 2 #min frequency
high_cut = 9.8 # max frequencu


template_list = glob.glob(output_folder_path+'/*')
len(template_list)

# Base directory to save files CC
base_directory_cc = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/'

# Variable folder name for 
folder_name = f'CC_{template_size}sec-tem_{start_date_process}-{end_date_process}'

# Full path
full_path = os.path.join(base_directory_cc, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(full_path):
    os.makedirs(full_path)
print(full_path)

   
# Parameters for the filter

b, a = butter(2, (low_cut, high_cut), 'bp', fs=fs)


## Buiding outputfiles and correlations for each template on the list

process_files_dos(file_list, template_list, chan_min, chan_max, channel_number, samples_per_file, b, a, full_path)

## Analisis of data

#creating timestamps

output_dir = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/h5_files_timestamps'
output_file_h5 = create_timestamps_h5(file_list, output_dir)

##  Timestamps_to utc
# Función para convertir timestamps a UTC

time_utc = convert_timestamps_to_utc(output_file_h5)

# Calculating MAD-threshold

def mad_func_shelly(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad

def process_folders(full_path, time_utc, matched_files_with_locations):
    try:
        # Obtener una lista de todas las carpetas en el directorio base
        folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]
    except Exception as e:
        print(f"Error listing folders in {full_path}: {e}")
        return

    # Inicializar una lista para almacenar los datos concatenados de cada carpeta
    concatenated_data_per_folder = []

    # Iterar sobre las carpetas y cargar los archivos .npy
    for folder in folders:
        folder_path = os.path.join(full_path, folder)
        try:
            npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
            concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            continue

    # Calcular MAD para cada carpeta y definir los umbrales
    mads_per_folder = {}
    thresholds_per_folder = {}
    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        try:
            mad = mad_func_shelly(folder_data)
            threshold = 18 * mad
            mads_per_folder[folder] = np.round(mad, decimals=3)
            thresholds_per_folder[folder] = np.round(threshold, decimals=3)
            print(f"MAD para la carpeta {folder}: {mads_per_folder[folder]}")
            print(f"Umbral para la carpeta {folder}: {thresholds_per_folder[folder]}")
        except Exception as e:
            print(f"Error calculating MAD for folder {folder}: {e}")

    # Crear archivos CSV para las detecciones
    output_directory = "template_{template_size}sec_csv_results"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    detection_times = []

    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        print(f"Processing folder {folder}")
        threshold = thresholds_per_folder[folder]
        indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]
        diff_indices = np.diff(indices_above_threshold)
        group_changes = np.where(diff_indices > 20)[0]
        detection_groups = np.split(indices_above_threshold, group_changes + 1)

        detection_times_folder = []

        # Buscar coincidencia con la carpeta
        matching_item = next((item for item in matched_files_with_locations if item['original_date'].replace(":", "-").replace(" ", "_") in folder), None)
        if matching_item:
            for group in detection_groups:
                if len(group) > 0:
                    first_detection_time_utc = time_utc[group[0]].strftime('%Y-%m-%d %H:%M:%S')
                    detection_times_folder.append({
                        'Detection Time (UTC)': first_detection_time_utc,
                        'Longitude': matching_item['longitude'],
                        'Latitude': matching_item['latitude']
                    })
            print(f"Matching item found for folder {folder}: {matching_item}")
        else:
            print(f"No matching item found for folder {folder}")

        print(f"Detections for folder {folder}: {detection_times_folder}")
        detection_times.extend(detection_times_folder)

        # Guardar en CSV
        df = pd.DataFrame(detection_times_folder)
        df.to_csv(os.path.join(output_directory, f'detections_{folder}.csv'), index=False)

    # Remover duplicados de todos los archivos CSV
    if detection_times:  # Ensure there are detection times to process
        all_detections = pd.DataFrame(detection_times)
        all_detections['Detection Time (UTC)'] = pd.to_datetime(all_detections['Detection Time (UTC)'])
        all_detections = all_detections.sort_values(by='Detection Time (UTC)')

        # Eliminar detecciones que estén dentro de los 5 segundos de cada una
        threshold_seconds = 10
        unique_detections = []
        previous_time = None

        for detection_time in all_detections['Detection Time (UTC)']:
            if previous_time is None or (detection_time - previous_time).total_seconds() > threshold_seconds:
                unique_detections.append(detection_time)
                previous_time = detection_time

        unique_detections_df = pd.DataFrame(unique_detections, columns=['Detection Time (UTC)'])
        unique_detections_df.to_csv(os.path.join(output_directory, 'unique_detections.csv'), index=False)

        print(f"Detections saved in {output_directory}")
    else:
        print("No detection times to process.")


# Encontrar archivos con fechas correspondientes
matched_files_with_locations = find_files_with_dates(event_df,base_path)
#print("Matched files with locations:")
#print(matched_files_with_locations)

# Convertir timestamps a UTC
time_utc = convert_timestamps_to_utc(output_file_h5)

# Procesar carpetas
process_folders(full_path, time_utc, matched_files_with_locations)




def process_folders_and_plot(full_path, fs, time_utc, output_plot_directory, found_files):
    try:
        folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]
    except Exception as e:
        print(f"Error listing folders in {full_path}: {e}")
        return

    concatenated_data_per_folder = []
    for folder in folders:
        folder_path = os.path.join(full_path, folder)
        try:
            npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
            concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))
        except Exception as e:
            print(f"Error processing folder {folder_path}: {e}")
            continue

    mads_per_folder = {}
    thresholds_per_folder = {}
    for folder, folder_data in zip(folders, concatenated_data_per_folder):
        try:
            mad = mad_func_shelly(folder_data)
            threshold = 18 * mad
            mads_per_folder[folder] = np.round(mad, decimals=3)
            thresholds_per_folder[folder] = np.round(threshold, decimals=3)
            print(f"MAD for folder {folder}: {mads_per_folder[folder]}")
            print(f"Threshold for folder {folder}: {thresholds_per_folder[folder]}")
        except Exception as e:
            print(f"Error calculating MAD for folder {folder}: {e}")

    if not os.path.exists(output_plot_directory):
        os.makedirs(output_plot_directory)

    for i, (folder, folder_data) in enumerate(zip(folders, concatenated_data_per_folder)):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_utc[:len(folder_data)], folder_data, label=f'Folder {folder}', color='blue', linestyle='-')
        ax.axhline(y=thresholds_per_folder[folder], color='red', linestyle='--', label='Threshold')
        ax.set_title(f'Template {folder}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation Value')
        ax.legend(loc='upper right')
        ax.grid(True)

        plot_filename = os.path.join(output_plot_directory, f'plot_{folder}.png')
        plt.savefig(plot_filename)
        plt.show()
        plt.close(fig)

    print(f"Plots saved in {output_plot_directory}")

process_folders_and_plot(full_path, fs, time_utc, output_plot_directory, found_files)
