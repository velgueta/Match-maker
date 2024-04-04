#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from TMfunctions import *



# In[2]:


#loading data to run, write here the path for your data.

file_list1 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-25_01*")
#file_list2 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-26*")
#file_list3 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-27*")
#file_list4 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-28*")
#file_list5 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-29*")
#file_list6 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-30*")
#file_list7 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-31*")

file_list = file_list1 #+ file_list2 + file_list3 + file_list4 + file_list5 + file_list6 + file_list7 

#organize your data by date
file_list.sort()


# In[3]:


# Path where are the templatesl

#template_list = glob.glob('/data/data4/veronica-scratch-rainier/templates-files/*') old templates de 1 segundo
template_list = glob.glob ('/data/data4/veronica-scratch-rainier/swarm_august2023/templates-files/templates-three-seconds/*')


# In[5]:


# Base directory to save files like CC
# base_directory = '/data/data4/veronica-scratch-rainier/test_corr' #prevoius correlations

# Base directory
base_directory = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/'

# Variable folder name
number = 6  #duration of the template (6 for test)
folder_name = f'CC_{number}sec-templates'

# Full path
full_path = os.path.join(base_directory, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(full_path):
    os.makedirs(full_path)
    print(f"Folder '{folder_name}' has been created in '{base_directory}'.")
else:
    print(f"Folder '{folder_name}' already exists in '{base_directory}'.")
    
    
corrs = "corrs"
#temp = "informacion de la ultima parte de cada template"

daystart = file_list[0]
dayend = file_list[-1]

# Extract date, hour, and minute portion from file path, excluding "new" and "decimator"
daystart_parts = os.path.basename(daystart).split('_')
daystart_date_time = "_".join(daystart_parts[2:4])  # Extract the date and time parts, excluding "new_decimator" and "UTC.h5"

dayend_parts = os.path.basename(dayend).split('_')
dayend_date_time = "_".join(dayend_parts[2:4])  # Extract the date and time parts, excluding "new_decimator" and "UTC.h5"

chan_min = 0 # from channel 0
chan_max = 3000  #to channel 3000
channel_number = chan_max -chan_min
low_cut1 = 2
hi_cut1 = 9.0
#fs=attrs['MaximumFrequency']*2
fs = 20
samples_per_file = 60*fs
b, a = butter(2, (low_cut1, hi_cut1), 'bp', fs=fs)


# # Buiding outputfiles and correlations for each template on the list

# In[6]:


start_time = time.perf_counter()

for i, file in tqdm(enumerate(file_list)):
    for j, tem in tqdm(enumerate(template_list)):
        try:
            # Load data file and template
            with h5py.File(file, "r") as f, h5py.File(tem, "r") as d:
                template = np.array(d['Acquisition/Raw[0]/RawData'][:, 0:-1])
                raw_data = np.array(f['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max-1])
                timestamps = np.array(f['Acquisition/Raw[0]/RawDataTime'])
                
                # Filter data
                data_filt = filtfilt(b, a, raw_data, axis=0)
                
                # Compute correlations
                corrs = window_and_correlate(template, data_filt)
                corrs2 = corrs.reshape((int(samples_per_file), channel_number-1))
                corrs3 = np.sum(corrs2, axis=1) / channel_number
                
                # Create output folder name
                folder_name_parts = os.path.splitext(os.path.basename(tem))[0].split('_')[:2]
                folder_name = '_'.join(folder_name_parts)
                folder_output = os.path.join(full_path, folder_name)
                
                # Create folder if it doesn't exist
                if not os.path.exists(folder_output):
                    os.mkdir(folder_output)
                
                # Save correlation results
                outfile_name = os.path.join(folder_output, f'corrs_{i}_.npy')
                np.save(outfile_name, corrs3)
        except Exception as e:
            print(f"Error processing file {file} with template {tem}: {e}. Skipping this file and moving to the next template.")
            continue

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"The code took {execution_time} seconds.")


# # Process the data

# After save all our CC's in full_path directory, we want to create some figures how this CC looks like over time and
# We want to get a final product (csv) with new catalog according to the thresold choosed = 10*MAD 
# 

# Loading timestamps

# In[7]:


#time .h5 for the swarm 
#for different data set, it has to be the time that is compared 

# Load the timestamps from the HDF5 file for the entire 7 days.
file_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/CC_1sec-templates/timestamps.h5'

# opening the file that containg the timestamps
with h5py.File(file_path, 'r') as f:
    timestamps_pt = np.array(f['timestamps'])

# Define the Pacific Timezone
pt_timezone = pytz.timezone('America/Los_Angeles')

# Convert timestamps from microseconds to seconds
timestamps_seconds = timestamps_pt / 1e6

# Convert timestamps to datetime objects in Pacific Time
datetime_objects_pt = [datetime.datetime.fromtimestamp(ts, pt_timezone) for ts in timestamps_seconds]

# Define the UTC Timezone
utc_timezone = pytz.timezone('UTC')

# Convert datetime objects to UTC
datetime_objects_utc = [dt_pt.astimezone(utc_timezone) for dt_pt in datetime_objects_pt]

# Redifining times variables
time_range = datetime_objects_utc


# In[8]:


# full_path is going to give you the CC data set for this run, also calcultin MAD so we can define the thresold

# Obtener una lista de todas las carpetas en el directorio base
folders = [folder for folder in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, folder))]

# Inicializar una lista para almacenar los datos concatenados por cada carpeta
concatenated_data_per_folder = []

# Iterar sobre las carpetas y cargar los archivos .npy
for folder in folders:
    folder_path = os.path.join(full_path, folder)
    npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
    concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))

# calcuting mad for each folder and for defining the thresold later
mads_per_folder = {}
for folder, folder_data in zip(folders, concatenated_data_per_folder):
    mad = np.median(np.abs(folder_data - np.median(folder_data)))
    mads_per_folder[folder] = mad
    print(f"MAD for folder {folder}: {mad}")
    
average_mad = np.mean(list(mads_per_folder.values()))

print(f"average mad {average_mad}")


# In[9]:


# Asumiendo que ya tienes definido time_range adecuadamente

# Plot the cc values for each folder

for i, folder_data in enumerate(concatenated_data_per_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(time_range[0:len(folder_data)], folder_data, label=f'Folder {folders[i]}', color='blue', linestyle='-')
    plt.axhline(y=0.02, color='red', linestyle='--') 
    plt.title(f'Template {folders[i]}')
    plt.xlabel('Time')
    plt.ylabel('Correlation Value')
    plt.legend()  # Agregar leyenda al gráfico
    plt.ylim([0, 0.5])  # Establecer límites del eje y de 0 a 0.5

    # Guardar la figura como PNG
    # Save the figure as PNG
    
    # The output folder is where 
    
    output_folder = '/data/data4/veronica-scratch-rainier/swarm_august2023/plot-TM-results/CC_plots-six-sec-templates'
    # Directorio de salida
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, f'corr_values_{folders[i]}.png')  # Ruta de salida
    plt.savefig(output_file_path, dpi=300)  # Guardar la figura como PNG
    plt.close()  # Cerrar la figura actual para evitar acumulación de figuras en memoria

#print("Proceso de creación de imágenes completado.")
print("Process completed")


# In[10]:


from datetime import datetime


# Making a catalog

# In[14]:


# Directorio donde se guardarán los archivos CSV
# Definir el directorio de salida
# Definir el directorio de salida
output_directory = "/home/velgueta/notebooks/RainierDas/csv_files"

# Iterar sobre cada carpeta
for folder in folders:
    # Ruta de la carpeta de salida
    folder_path = os.path.join(output_directory, folder)
    
    # Ruta del archivo de salida
    output_file_path = os.path.join(folder_path, f"detections_results_{folder}.csv")
    
    print("Intentando abrir el archivo:", output_file_path)  # Imprimir la ruta del archivo
    
    # Abre el archivo de salida en modo lectura con la biblioteca CSV
    with open(output_file_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=',')

        
        # Busca el valor repetido para el umbral especificado
        for row in reader:
            threshold, _, detection_time_utc = row
            if float(threshold) == target_threshold:
                if target_threshold not in detection_times_per_threshold:
                    detection_times_per_threshold[target_threshold] = set()
                detection_times_per_threshold[target_threshold].add(detection_time_utc)

# Diccionario para almacenar los valores únicos por umbral y tolerancia de tiempo
unique_detection_times_per_threshold = {}

# Itera sobre cada umbral y sus valores de tiempo
for threshold, detection_times in detection_times_per_threshold.items():
    unique_detection_times = set()
    for detection_time in detection_times:
        detection_time_obj = datetime.strptime(detection_time, '%Y-%m-%d_%H.%M.%S')
        # Verifica si hay valores dentro de una tolerancia de 3 segundos
        is_unique = True
        for unique_time in unique_detection_times:
            time_difference = abs((detection_time_obj - unique_time).total_seconds())
            if time_difference <= 4:
                is_unique = False
                break
        if is_unique:
            unique_detection_times.add(detection_time_obj)
    unique_detection_times_per_threshold[threshold] = unique_detection_times

# Guarda los valores únicos en un nuevo archivo CSV llamado "draftcatalog_3sec.csv"
draftcatalog_file_path = "draftcatalog_6sec.csv"

with open(draftcatalog_file_path, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for threshold, unique_detection_times in unique_detection_times_per_threshold.items():
        for detection_time in unique_detection_times:
            writer.writerow([threshold, detection_time.strftime('%Y-%m-%d_%H.%M.%S')])

print("Valores únicos guardados en 'draftcatalog_6sec.csv'.")


# In[20]:


StopIteration                             Traceback (most recent call last)
Cell In [19], line 24
     21 reader = csv.reader(file, delimiter=',')
     23 # Ignora la primera fila (encabezado)
---> 24 next(reader)
     26 # Busca el valor repetido para el umbral especificado
     27 for row in reader:

StopIteration:


# In[ ]:


from datetime import datetime, timedelta


# In[ ]:


# Define the function to parse a date string into a datetime object
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d_%H.%M.%S')
    except ValueError:
        return None

# Define the time difference allowed (5 seconds)
time_difference = timedelta(seconds=6)

# Function to read dates and values from a file and return a list of tuples containing both
def read_dates_and_values_from_file(file_path):
    dates_and_values = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # Skip header line
        for row in reader:
            if len(row) >= 2:  # Ensure there are at least two elements
                value = row[0]  # Extract the value from the first column
                date_str = row[-1]  # Extract the date from the last column
                date = parse_date(date_str)
                if date is not None:
                    dates_and_values.append((value, date))
    return dates_and_values

# Read dates and values from the converted_dates_and_values_rainier_5days.csv file
converted_dates_and_values = read_dates_and_values_from_file('converted_dates_and_values_rainier_5days.csv')

# Initialize lists to store common dates and unmatched dates
common_dates = []
unmatched_dates = []

# Read dates from the draftcatalog_3sec.csv file and compare with converted dates
with open('draftcatalog_6sec.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)  # Skip header line
    for row in reader:
        threshold_value, result_date_str = row[:2]
        result_date = parse_date(result_date_str)  # Convert result date to a datetime object
        if result_date is not None:  # Check if result_date is not None
            # Check if the result_date has a match in converted_dates_and_values
            matched = False
            for value, converted_date in converted_dates_and_values:
                if abs(result_date - converted_date) <= time_difference:
                    common_dates.append((threshold_value, result_date, value))
                    matched = True
                    break
            if not matched:
                unmatched_dates.append((threshold_value, result_date))

# Write common and unmatched dates to the new output file
output_file_path = os.path.join(output_directory, 'matched_results_finalcatalog.csv')

# Write common and unmatched dates to the new output file
with open(output_file_path, 'w', newline='') as f:

    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Threshold", "Result Date", "ID Value", "Match Status"])
    for threshold_value, result_date, id_value in common_dates:
        writer.writerow([threshold_value, result_date.strftime('%Y-%m-%d_%H.%M.%S'), id_value, "Matched"])
    for threshold_value, result_date in unmatched_dates:
        writer.writerow([threshold_value, result_date.strftime('%Y-%m-%d_%H.%M.%S'), "X", "Unmatched"])

print("Output file 'matched_results_finalcatalog.csv' has been generated.")


# Saving the of matched_results_finalcatalog figures in png format 
# 

# In[ ]:


# output_file_path is where our final catalog it is

txt_file_path = output_file_path
files_folder_path = '/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds'
plots_folder = '/data/data4/veronica-scratch-rainier/swarm_august2023/plot-TM-results/plots-newcatalog'

# Regular expression pattern to search for dates in the correct format
date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})')

# Extract information from output.txt
result_dates = []
thresholds = []
id_values = []
match_statuses = []

with open(txt_file_path, 'r') as txt_file:
    for line in txt_file:
        columns = line.strip().split('\t')
        threshold = columns[0]
        result_date = columns[1]
        id_value = columns[2]
        match_status = columns[3]
        result_dates.append(result_date)
        thresholds.append(threshold)
        id_values.append(id_value)
        match_statuses.append(match_status)

# Define the threshold to search

threshold_to_search = average_mad*10

# Create the folder for plots if it doesn't exist
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Create a folder to hold all the plots
all_plots_folder = os.path.join(plots_folder, f'{threshold_to_search}_plots')
if not os.path.exists(all_plots_folder):
    os.makedirs(all_plots_folder)

# Iterate over the result dates and search for corresponding files with the specific threshold

for i, result_date in enumerate(result_dates):
    threshold = thresholds[i]
    id_value = id_values[i]
    match_status = match_statuses[i]
    match = date_pattern.match(result_date)
    if match:
        date_part = match.group(1)
        hour_part = match.group(2)
        minute_part = match.group(3)
        date_to_search = f"{date_part}_{hour_part}.{minute_part}"
        if threshold == threshold_to_search:
            found = False
            for file_name in os.listdir(files_folder_path):
                if date_to_search in file_name:
                    print(f"Found a file with the searched date: {file_name}")
                    found = True

                    # Process the found file
                    chan_min = 0
                    chan_max = -1
                    data_file_path = os.path.join(files_folder_path, file_name)
                    data_file = h5py.File(data_file_path, 'r')
                    this_data = np.array(data_file['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max])
                    this_time = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])
                    attrs = dict(data_file['Acquisition'].attrs)
                    data_file.close()

                    low_cut1 = 2
                    hi_cut1 = 9
                    fs = 20
                    b, a = butter(2, (low_cut1, hi_cut1), 'bp', fs=fs)
                    data_filt = filtfilt(b, a, this_data, axis=0)

                    date_format = mdates.DateFormatter('%H:%M:%S')
                    x_lims = mdates.date2num(this_time)
                    x_max = data_filt.shape[1] * attrs['SpatialSamplingInterval'] / 1000
                    dx = x_max / data_filt.shape[1]

                    fig, ax = plt.subplots(figsize=(20, 10))
                    plt.imshow(data_filt.T, cmap='seismic', aspect='auto', vmin=-0.05, vmax=0.05,
                               extent=[x_lims[0], x_lims[-1], x_max, 0])
                    plt.xlabel("Time UTC", fontsize=25)
                    plt.ylabel("Optical distance (km)", fontsize=25)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.xaxis_date()

                    # Set font size for time and channel axis labels
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)

                    # Set title including threshold, ID value, and match status
                    plt.title(f'Threshold: {threshold}, ID: {id_value}, Match Status: {match_status}', fontsize=20)

                    # Save the plot in the all_plots_folder
                    full_path = os.path.join(all_plots_folder, f'{file_name}.png')
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    break

            if not found:
                print(f"No matching file found for the searched date: {date_to_search}")

