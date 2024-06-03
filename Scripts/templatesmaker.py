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
import csv
import re
import matplotlib.dates as mdates


# Este archivo crea templates apartir de una base de datos importadas desde un archivo csv, luego busca esos datos en 
# el DAS data y selecciona una ventana de tiempo apropiada (6 sec, 10 sec etc, y los guarda en cierto folder )

# In[2]:


# save the templates made here, /data/data4/veronica-scratch-rainier/swarm_august2023/test-template-maker
# vamos a extraer las fechas desde dates_test.csv, the format is id number of the event and hypocenter date, like this
# 61953701,2023-08-27_10.10.23
# vamos a hacer un template de 10 sec, cada archivo tiene 1200 puntos, por los que 200 puntos
# vamos a solo usar 0 a 2500 channels que son los que considero buenos.
# Aquí se encuentran los files a buscar file_list = glob.glob("/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds/)


# In[3]:


# Parameters
chan_min = 0
chan_max = 2500


# In[4]:


# Rutas de los directorios y archivo CSV
csv_file_path = 'dates_test.csv'
files_folder_path = '/data/fast1/veronica-scratch-rainier-downsampling/drive1_ds'
output_folder_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/test-template-maker/twosec/raw'
output_folder_path_filter = '/data/data4/veronica-scratch-rainier/swarm_august2023/test-template-maker/twosec/filtering'

# Expresión regular para buscar fechas en el formato correcto (incluyendo hora y minutos)
date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2})')

# Lista para almacenar las fechas del archivo CSV
original_dates = []

# Leer las fechas del archivo CSV
with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        date_with_seconds = row[1]
        original_dates.append(date_with_seconds)

# Lista para almacenar los nombres de los archivos encontrados
found_files = []

# Iterar sobre las fechas originales y buscar archivos correspondientes con la fecha específica
for date_to_search in original_dates:
    match = date_pattern.match(date_to_search)
    if match:
        formatted_date = match.group(1)
        for file_name in os.listdir(files_folder_path):
            if formatted_date in file_name:
                found_files.append(file_name)
                print(f"Found a file with the searched date: {file_name}")
                
for file_name in found_files:
    data_file_path = os.path.join(files_folder_path, file_name)
    with h5py.File(data_file_path, 'r') as data_file:
        this_data = np.array(data_file['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max])
        this_time = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])
        
# Imprimir los nombres de los archivos encontrados
#print("Archivos encontrados:")
#for file_name in found_files:
#    print(file_name)


# In[5]:


from datetime import datetime, timedelta

# Procesar los archivos encontrados
for file_name in found_files:
    data_file_path = os.path.join(files_folder_path, file_name)
    output_file_path = os.path.join(output_folder_path, file_name)
    with h5py.File(data_file_path, 'r') as data_file:
        # Obtener los datos y el tiempo
        data = np.array(data_file['Acquisition/Raw[0]/RawData'][:, chan_min:chan_max])
        time = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])
        #print(time)
        
        # Convertir las marcas de tiempo a objetos de datetime UTC
        time_utc = [datetime.utcfromtimestamp(ts / 1000000) for ts in time]
        
        # Encontrar la marca de tiempo correspondiente en el tiempo UTC para la fecha de inicio y fin
        date_to_search = original_dates[found_files.index(file_name)]
        start_date = datetime.strptime(date_to_search, "%Y-%m-%d_%H.%M.%S") + timedelta(seconds=2)
        end_date = start_date + timedelta(seconds=2)  # Añadir 4,8 etc segundos al inicio
        
        # Convertir las marcas de tiempo de inicio y fin a UTC
        start_date_utc = start_date - (timedelta(hours=start_date.utcoffset().total_seconds() / 3600) if start_date.utcoffset() else timedelta(0))
        end_date_utc = end_date - (timedelta(hours=end_date.utcoffset().total_seconds() / 3600) if end_date.utcoffset() else timedelta(0))
        
        # Encontrar los índices correspondientes en el tiempo para las fechas de inicio y fin en UTC
        start_index = np.argmin(np.abs(np.array(time_utc) - start_date_utc))
        end_index = np.argmin(np.abs(np.array(time_utc) - end_date_utc))

        # Cortar los datos en el rango de tiempo deseado
        data_cut = data[start_index:end_index]
        time_cut = time[start_index:end_index]

        # Guardar los datos cortados en un nuevo archivo HDF5
        with h5py.File(output_file_path, 'w') as output_file:
            output_file.create_dataset('Acquisition/Raw[0]/RawData', data=data_cut)
            output_file.create_dataset('Acquisition/Raw[0]/RawDataTime', data=time_cut)


# In[6]:


# Procesar los archivos encontrados
for file_name in found_files:
    data_file_path = os.path.join(output_folder_path, file_name)  # Usar la carpeta de salida
    with h5py.File(data_file_path, 'r') as data_file:
        # Obtener los datos y el tiempo cortados
        data_cut = np.array(data_file['Acquisition/Raw[0]/RawData'])
        time_cut = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])

        # Convertir las marcas de tiempo a objetos de datetime UTC
        time_utc = [datetime.utcfromtimestamp(ts / 1000000) for ts in time_cut]

        # Configurar la figura y los ejes
        fig, ax = plt.subplots(figsize=(20, 10))

        # Graficar los datos
        plt.imshow(data_cut.T, cmap='seismic', aspect='auto', vmin=-0.05, vmax=0.05,
                   extent=[mdates.date2num(time_utc[0]), mdates.date2num(time_utc[-1]), data_cut.shape[1], 0])

        # Configurar etiquetas y formato de fecha en el eje x
        plt.xlabel("Tiempo UTC", fontsize=25)
        plt.ylabel("Distancia óptica (km)", fontsize=25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis_date()

        # Mostrar la figura
        plt.show()


# In[7]:


#filtering these files

from datetime import datetime, timedelta

# Define constants for filtering
low_cut1 = 2
hi_cut1 = 9
fs = 20

# Iterate over files in the output folder
for file_name in os.listdir(output_folder_path):
    if file_name.endswith('.h5'):  # Process only HDF5 files
        # Construct paths
        data_file_path = os.path.join(output_folder_path, file_name)
        output_file_path_filtering = os.path.join(output_folder_path_filter, file_name)
        
        # Open the HDF5 file
        with h5py.File(data_file_path, 'r') as data_file:
            # Extract data and time
            this_data = np.array(data_file['Acquisition/Raw[0]/RawData'][:, :])
            #len(this_data)# Assuming you want all channels
            this_time = np.array(data_file['Acquisition/Raw[0]/RawDataTime'])
        
        # Apply Butterworth bandpass filter
        
            b,a = butter(2,(low_cut1,hi_cut1),'bp',fs=fs)
            data_filt = filtfilt(b,a,this_data,axis=0)
        
        # Save the filtered data into a new HDF5 file
        with h5py.File(output_file_path_filtering, 'w') as output_file:
            output_file.create_dataset('Acquisition/Raw[0]/RawData', data=data_filt)
            output_file.create_dataset('Acquisition/Raw[0]/RawDataTime', data=this_time)


# In[8]:


for file_name in os.listdir(output_folder_path_filter):
    if file_name.endswith('.h5'):  # Procesar solo archivos HDF5
        # Construir la ruta al archivo HDF5 filtrado
        filtered_file_path = os.path.join(output_folder_path_filter, file_name)
        
        # Abrir el archivo HDF5 filtrado
        with h5py.File(filtered_file_path, 'r') as filtered_file:
            # Leer los datos filtrados y el tiempo desde el archivo HDF5
            filtered_data = np.array(filtered_file['Acquisition/Raw[0]/RawData'])
            time_data = np.array(filtered_file['Acquisition/Raw[0]/RawDataTime'])
        
        # Mostrar los datos filtrados utilizando imshow
        plt.figure(figsize=(12, 6))
        plt.imshow(filtered_data.T, cmap='seismic', aspect='auto', vmin=-0.05, vmax=0.05,
                   extent=(time_data[0], time_data[-1], 0, filtered_data.shape[1]))
        plt.colorbar(label='Amplitude')
        plt.title(f'Filtered Data: {file_name}')
        plt.xlabel('Time')
        plt.ylabel('Channel')
        plt.show()


# In[ ]:




