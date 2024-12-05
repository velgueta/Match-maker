import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

# Define el directorio base
base_directory_cc = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/'

# Nombre variable de la carpeta
folder_name_v1 = 'CC_2sec-tem_2023-08-27_10.00-2023-08-27_12.59/' 
# Ruta completa
full_path = os.path.join(base_directory_cc, folder_name_v1)

# Crea la carpeta si no existe
if not os.path.exists(full_path):
    os.makedirs(full_path)
print(full_path)

def mad_func_shelly(arr):
    """Desviación Absoluta Mediana: Usando la formulación en Li y Zhan 2018."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def calculate_detection_sig(folder_data):
    """Calcula la significancia de la detección para los datos de una carpeta dada."""
    median = np.median(folder_data)
    mad = mad_func_shelly(folder_data)
    detection_sig = (folder_data - median) / mad
    return detection_sig, mad

def plot_histogram(ax, detection_sig, folder_name):
    """Grafica el histograma de la significancia de detección."""
    ax.hist(detection_sig, bins=1000, range=(0, 500), alpha=0.75, color='blue', edgecolor='blue', linewidth=2.5)
    ax.set_yscale('log')  # Establece el eje y en escala logarítmica
    ax.set_xlabel('Detection Significance', fontsize=14)
    ax.set_ylabel('Counts (log scale)', fontsize=14)
    ax.grid(True, which="both", ls="--")

def plot_time_series(ax, time_utc, folder_data, mad, folder_name):
    """Grafica la serie temporal del valor de correlación sobre MAD."""
    ax.plot(time_utc, folder_data / mad, label=f'Template {folder_name}', color='blue', linestyle='-')
    ax.set_title(f'Template {folder_name}')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Correlation Value/MAD')
    ax.legend(loc='upper right')
    ax.grid(True)

def process_single_folder(full_path, folder_name, output_plot_directory, h5_file_path):
    folder_path = os.path.join(full_path, folder_name)
    try:
        npy_files = [np.load(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith('.npy')]
        folder_data = np.concatenate(npy_files, axis=0)

        # Leer los timestamps del archivo .h5
        with h5py.File(h5_file_path, 'r') as h5_file:
            time_utc = np.array(h5_file['timestamps'])
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return
    
    detection_sig, mad = calculate_detection_sig(folder_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    plot_time_series(ax1, time_utc[:len(folder_data)], folder_data, mad, folder_name)
    plot_histogram(ax2, detection_sig, folder_name)

    plot_filename = os.path.join(output_plot_directory, f'plot_{folder_name}.png')
    plt.savefig(plot_filename)
    plt.show()
    #plt.close(fig)

# Uso de ejemplo
output_plot_directory = os.path.join(full_path, 'plots')
if not os.path.exists(output_plot_directory):
    os.makedirs(output_plot_directory)

folder_name = '2023-08-27_10.10.00'  # Reemplaza con el nombre de la carpeta que deseas procesar
h5_file_path = '/data/data4/veronica-scratch-rainier/swarm_august2023/results_CC_TMA/h5_files_timestamps/timestamps_2023-08-27_10.00.00_2023-08-27_12.59.00.h5'  # Ruta al archivo .h5 que contiene los timestamps
process_single_folder(full_path, folder_name, output_plot_directory, h5_file_path)






