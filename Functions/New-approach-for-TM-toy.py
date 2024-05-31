#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


#loading data to run

file_list1 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-25*")
file_list2 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-26*")
file_list3 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-27*")
file_list4 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-28*")
file_list5 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-29*")
file_list6 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-30*")
file_list7 = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/new_decimator_2023-08-31*")

file_list = file_list1 + file_list2 + file_list3 + file_list4 + file_list5 + file_list6 + file_list7 

file_list.sort()


# In[ ]:


#loading templates

#template_list = glob.glob('/data/data4/veronica-scratch-rainier/templates-files/*')
template_list = glob.glob ('/data/data4/veronica-scratch-rainier/templates-files/test-newtemplates/*')


# In[ ]:


# Base directory to save files
#base_directory = '/data/data4/veronica-scratch-rainier/test_corr'
base_directory = '/data/data4/veronica-scratch-rainier/test_corr-secondround'
#the folder contain in the name_numbers date of the template used

#base_directory = '/home/velgueta/notebooks/RainierDas' #test

corrs = "corrs"
#temp = "informacion de la ultima parte de cada template"

daystart = file_list[0]
dayend = file_list[-1]

# Extract date, hour, and minute portion from file path, excluding "new" and "decimator"
daystart_parts = os.path.basename(daystart).split('_')
daystart_date_time = "_".join(daystart_parts[2:4])  # Extract the date and time parts, excluding "new_decimator" and "UTC.h5"

dayend_parts = os.path.basename(dayend).split('_')
dayend_date_time = "_".join(dayend_parts[2:4])  # Extract the date and time parts, excluding "new_decimator" and "UTC.h5"

chan_min = 0#1000
chan_max = 3000#2500
channel_number = chan_max -chan_min
low_cut1 = 2
hi_cut1 = 9.0
#fs=attrs['MaximumFrequency']*2
fs = 20
samples_per_file = 60*fs
b, a = butter(2, (low_cut1, hi_cut1), 'bp', fs=fs)


# # Buiding outputfiles and correlations for each template on the list

# In[ ]:


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
                folder_output = os.path.join(base_directory, folder_name)
                
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


# In[ ]:


#time vector for plotting correlations

#time steps
timesteps = len(file_list)* samples_per_file
# Extraer las fechas de inicio y fin del nombre de la carpeta de salida

#just formatting

# Convertir las cadenas de tiempo a objetos datetime
start_datetime = pd.to_datetime(daystart_date_time, format="%Y-%m-%d_%H.%M.%S")
end_datetime = pd.to_datetime(dayend_date_time, format="%Y-%m-%d_%H.%M.%S")

#final vector
time_range = pd.date_range(start=start_datetime, end=end_datetime, periods=timesteps)


# In[ ]:


#for 2 days, 3 templates took 41 min
# Ruta base del directorio donde se encuentran las carpetas con los archivos .npy



# Obtener nombres de todas las carpetas dentro del directorio base
folders = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

# Función para cargar los archivos .npy de una carpeta y devolverlos como una lista
def load_npy_files(folder):
    npy_files = []
    for file in os.listdir(folder):
        if file.endswith('.npy'):
            npy_file = np.load(os.path.join(folder, file))
            npy_files.append(npy_file)
    return npy_files


# In[ ]:


# Inicializar una lista para almacenar los datos concatenados por cada carpeta
concatenated_data_per_folder = []

# Iterar sobre las carpetas y cargar los archivos .npy
for folder in folders:
    folder_path = os.path.join(base_directory, folder)
    npy_files = load_npy_files(folder_path)
    concatenated_data_per_folder.append(np.concatenate(npy_files, axis=0))

# Graficar los datos para cada carpeta
fig, axs = plt.subplots(len(folders), 1, figsize=(10, 5*len(folders)))

for i, folder_data in enumerate(concatenated_data_per_folder):
    axs[i].plot(time_range, folder_data, label=f'Folder {folders[i]}', color='blue', linestyle='-')
    axs[i].set_title(f'Template {folders[i]}')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Correlation Value')
    axs[i].legend()  # Add legend to each subplot
    axs[i].set_ylim([0,0.2])  # Set y-axis limits from 0 to 0.5


plt.tight_layout()
plt.grid()
plt.show()


# # add thresold and detections stuff

# In[ ]:



# Define the thresholds
thresholds = [0.5, 0.1, 0.05, 0.04]

# Iterate over each folder
for folder, folder_data in zip(folders, concatenated_data_per_folder):
    print(f"Threshold detections for folder {folder}:")
    
    # Iterate over each threshold
    for threshold in thresholds:
        # Find indices where values exceed the threshold
        indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]

        # Find differences between consecutive indices
        diff_indices = np.diff(indices_above_threshold)

        # Find changes in differences indicating a new group
        group_changes = np.where(diff_indices > 40)[0]

        # Group indices into contiguous lists
        detection_groups = np.split(indices_above_threshold, group_changes + 1)

        # Count the number of detections for the current threshold
        count_above_threshold = len(detection_groups)

        # Print the result for the current threshold
        print(f"  Detections for threshold {threshold}: {count_above_threshold}")


# In[ ]:


import matplotlib.pyplot as plt

# Define los umbrales
thresholds = [0.5, 0.1, 0.05, 0.04]

# Itera sobre cada carpeta
for i, folder_data in enumerate(concatenated_data_per_folder):
    # Inicializa una lista para almacenar el número de detecciones para cada umbral
    detections_counts = []
    
    # Itera sobre cada umbral
    for threshold in thresholds:
        # Encuentra los índices donde los valores superan el umbral
        indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]

        # Encuentra las diferencias entre índices consecutivos
        diff_indices = np.diff(indices_above_threshold)

        # Encuentra los cambios en las diferencias que indican un nuevo grupo
        group_changes = np.where(diff_indices > 40)[0]

        # Agrupa los índices en listas contiguas
        detection_groups = np.split(indices_above_threshold, group_changes + 1)

        # Cuenta el número de detecciones para el umbral actual
        count_above_threshold = len(detection_groups)
        
        # Añade el número de detecciones a la lista
        detections_counts.append(count_above_threshold)
    
    # Grafica los datos para la carpeta actual
    plt.figure(figsize=(8, 6))
    plt.scatter(thresholds, detections_counts, marker='o')
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Detections')
    plt.title(f'Number of Detections vs Threshold Value for Folder {folders[i]}')
    plt.grid(True)
    plt.show()


# In[ ]:


#saving the results in txt file
import datetime

# Define los umbrales
thresholds = [0.5, 0.1, 0.05, 0.04, 0.035, 0.03]

# Ruta del archivo de salida
output_file_path_TM = "detections_results.txt"

# Abre el archivo de salida en modo escritura
with open(output_file_path_TM, 'w') as file:
    # Escribe los encabezados
    file.write("Folder\tThreshold\tNumber of Detections\tDetection Times (UTC)\n")
    
    # Itera sobre cada carpeta
    for i, folder_data in enumerate(concatenated_data_per_folder):
        # Itera sobre cada umbral
        for threshold in thresholds:
            # Encuentra los índices donde los valores superan el umbral
            indices_above_threshold = np.where(np.abs(folder_data) > threshold)[0]
            diff_indices = np.diff(indices_above_threshold)
            group_changes = np.where(diff_indices > 40)[0]
            detection_groups = np.split(indices_above_threshold, group_changes + 1)
            
            # Itera sobre cada grupo de detección
            for group in detection_groups:
                # Toma solo la primera fecha en cada grupo
                first_detection_time_utc = datetime.datetime.utcfromtimestamp(time_range[group[0]].timestamp()).strftime('%Y-%m-%d_%H.%M.%S')
                
                # Escribe los datos en una fila separados por tabulaciones
                file.write(f"{folders[i]}\t{threshold}\t{len(detection_groups)}\t{first_detection_time_utc}\n")

# Imprime el mensaje de confirmación
print(f"The results have been saved in {output_file_path_TM}")


# In[ ]:


time_range


# In[ ]:




