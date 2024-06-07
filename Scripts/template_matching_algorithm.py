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
from getanalisisfiles import *




#loading data to run, write here the path for your data.

base_path = "/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds"
start_date = "2023-08-25_00"
end_date = "2023-08-25_00"

# using the fuction get_file_list 
file_list = get_file_list(base_path,start_date,end_date)


## templates
template_list = glob.glob ('/data/data4/veronica-scratch-rainier/swarm_august2023/test-template-maker/test_4sec_templates/filtering/*')

# Base directory to save files CC
base_directory_cc = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/'

# Variable folder name
template_size = 21  #20 #21 frankesitein, duration of the template (6 for test)
folder_name = f'CC_{template_size}sec-templates'

# Full path
full_path = os.path.join(base_directory_cc, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(full_path):
    os.makedirs(full_path)
print(full_path)
    

## Parameters for the filter

chan_min = 0     # from channel 0
chan_max = 2500  #to channel 3000
channel_number = chan_max -chan_min
low_cut1 = 2
hi_cut1 = 9.0
#fs=attrs['MaximumFrequency']*2
fs = 20
samples_per_file = 60*fs
b, a = butter(2, (low_cut1, hi_cut1), 'bp', fs=fs)


## Buiding outputfiles and correlations for each template on the list

process_files(file_list, template_list, chan_min, chan_max, channel_number, samples_per_file, b, a, full_path)


## Analisis of data
#creating timestamps for the swarm window
# input

output_dir = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/h5_files_timestamps'

#
output_file_h5 = create_timestamps_h5(file_list, output_dir)

##
#Timestamps_to utc

time_utc = convert_timestamps_to_utc(output_file_h5)

##
#this fuction is not working yet, it does not calculate the MAD correctly
process_folders(full_path, fs=20)

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

#print(f"average mad {average_mad}")


# Plots and Saving CC plots with thresold in red line

thresold = np.round(average_mad*10, decimals=3)

#Insert directory here: 
base_directory = '/data/data4/veronica-scratch-rainier/swarm_august2023/plot-TM-results/'

# Variable folder name
long_template = 4  #duration of the template (6 for test)
folder_name = f'CC_{long_template}sec-templates'

    # Full path
output_folder_figures = os.path.join(base_directory, folder_name)

# Create folder if it doesn't exist
if not os.path.exists(output_folder_figures):
    os.makedirs(output_folder_figures)
    print(f"Folder '{folder_name}' has been created in '{base_directory}'.")
else:
    print(f"Folder '{folder_name}' already exists in '{base_directory}'.")

for i, folder_data in enumerate(concatenated_data_per_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(time_range[0:len(folder_data)], folder_data, label=f'Folder {folders[i]}', color='blue', linestyle='-')
    plt.axhline(y=thresold, color='red', linestyle='--') 
    plt.title(f'Template {folders[i]}')
    plt.xlabel('Time')
    plt.ylabel('Correlation Value')
    plt.legend()  # Agregar leyenda al gráfico
    #plt.ylim([0, 0.5])  # Establecer límites del eje y de 0 a 0.5

    # The output folder is where 
    
    output_file_path = os.path.join(output_folder_figures, f'corr_values_{folders[i]}.png')  # Ruta de salida
    plt.savefig(output_file_path, dpi=300)  # Guardar la figura como PNG
    plt.close()  # Cerrar la figura actual para evitar acumulación de figuras en memoria

#print("Proceso de creación de imágenes completado.")
print("Process completed")





from datetime import datetime


# Making a catalog




# For each folder a csv files that save the detections for the thresold chose it

output_directory = '/home/velgueta/notebooks/RainierDas/csv_files'

# Check and create the main output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over each folder and concatenated data per folder

for folder, folder_data in zip(folders, concatenated_data_per_folder):
    # Create output directory for this folder if it doesn't exist
    folder_output_directory = os.path.join(output_directory, folder)
    if not os.path.exists(folder_output_directory):
        os.makedirs(folder_output_directory)
    
    # Path for the output file for this folder
    output_file_path = os.path.join(folder_output_directory, f"detections_results_{folder}.csv")
    
    # Open the output file in write mode using CSV
    with open(output_file_path, 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile, delimiter=',')
        
        # Write headers in the CSV file
        csv_writer.writerow(["Threshold", "Number of Detections", "First Detection Time (UTC)"])
        
        # Define the threshold (for example, average_mad * 10)
        threshold = np.round(average_mad * 18, decimals=3)
        
        # Iterate over each detection threshold (example with a single threshold)
        # If you want multiple thresholds, use a list of thresholds and a for loop
        indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]
        diff_indices = np.diff(indices_above_threshold)
        group_changes = np.where(diff_indices > 40)[0]
        detection_groups = np.split(indices_above_threshold, group_changes + 1)
        
        # Iterate over each detection group
        for group in detection_groups:
            if len(group) > 0:
                # Get the first detection time in UTC format
                first_detection_time_utc = time_range[group[0]].strftime('%Y-%m-%d_%H.%M.%S')
                # Write a row in the CSV file
                csv_writer.writerow([threshold, len(group), first_detection_time_utc])
            else:
                print("Empty group found for threshold:", threshold)

# Confirmation message
print(f"Results have been saved in {output_directory}")





## cleaning the repeat values for folder, having a unique ouput file with the dates

from datetime import datetime, timedelta

# Directorio donde se encuentran las carpetas con los archivos CSV
input_directory = "/home/velgueta/notebooks/RainierDas/csv_files"

# Directorio donde se guardará el archivo CSV final con fechas únicas entre carpetas
output_directory = "/home/velgueta/notebooks/RainierDas/Catalogs-csv"

# Duración de tiempo para la detección (en segundos)
template_long = 4

# Umbral de tolerancia en segundos para considerar que las fechas son "repetidas"
tolerance_seconds = 5

# Diccionario para almacenar las fechas de detección únicas por carpeta
unique_detection_dates_by_folder = {}

# Función para verificar si una fecha es única respecto a las fechas de otras carpetas
def is_unique_between_folders(all_detection_dates, new_date, tolerance_seconds):
    for existing_date in all_detection_dates:
        if abs((new_date - existing_date).total_seconds()) <= tolerance_seconds:
            return False
    return True

# Iterar sobre cada carpeta en el directorio de entrada
for folder in os.listdir(input_directory):
    folder_path = os.path.join(input_directory, folder)
    
    # Ignorar elementos que no son directorios
    if not os.path.isdir(folder_path):
        continue
    
    # Conjunto para almacenar las fechas de detección únicas dentro de la carpeta actual
    unique_detection_dates = set()
    
    # Iterar sobre los archivos CSV dentro de la carpeta
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            
            # Abrir el archivo CSV y leer las fechas de detección
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Saltar la primera fila (encabezado)
                
                for row in reader:
                    _, _, detection_time_str = row[0], row[1], row[2]
                    detection_time = datetime.strptime(detection_time_str, '%Y-%m-%d_%H.%M.%S')
                    
                    # Verificar si la fecha es única entre carpetas dentro del rango de tolerancia
                    if is_unique_between_folders(unique_detection_dates, detection_time, tolerance_seconds):
                        unique_detection_dates.add(detection_time)
    
    # Guardar las fechas de detección únicas en el diccionario por carpeta
    unique_detection_dates_by_folder[folder] = unique_detection_dates

# Nombre del archivo CSV final y ruta de salida
output_file_name = f"unique_detection_dates_{template_long}s.csv"
output_file_path_catalog = os.path.join(output_directory, output_file_name)

# Escribir las fechas únicas entre carpetas en el archivo CSV final
with open(output_file_path_catalog, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Unique Detection Dates"])  # Escribir encabezado
    
    # Diccionario para todas las fechas de detección entre carpetas
    all_detection_dates = set()
    
    # Iterar sobre cada carpeta y sus fechas únicas
    for folder, detection_dates in unique_detection_dates_by_folder.items():
        # Iterar sobre cada fecha única de la carpeta actual
        for date in detection_dates:
            # Verificar si la fecha es única entre todas las fechas de detección
            if is_unique_between_folders(all_detection_dates, date, tolerance_seconds):
                all_detection_dates.add(date)
                # Escribir la fecha única en el archivo CSV final
                writer.writerow([date.strftime('%Y-%m-%d_%H.%M.%S')])

print(f"Fechas únicas entre carpetas guardadas en '{output_file_path_catalog}'.")


# Since Here is not working accurately! working in the debugging part now!

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
with open('unique_detection_dates_4s', 'r') as f:
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





# Comparing 


# Saving the of matched_results_finalcatalog figures in png format 
# 




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

