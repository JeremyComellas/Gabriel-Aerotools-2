# Código utilizado para generar plots a partir de los files .toml, que contienen guardados los resultados de las iteracioines de parámetros (contrast_threshold, gridSize, sigma... etc).
# Básicamente, el toml file contiene los datos utilizados para hacer los gráficos que muestran la variación de los indicadores (numero de matches, match size.... etc)
# Cambiar el nombre del toml file que importa el programa et voila


import toml
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the TOML file
file_path = "Data_processed\tests_find_best_algo\Stitched_Images_6_6__int_float\output_trees_2024-08-13_16-59-19.toml"


with open(file_path, 'r') as file:
    data = toml.load(file)
# Convert string values to floats where necessary
for key, value in data.items():
    data[key] = [float(x) for x in value]



def get_picture_filenames(folder_path):
            # List of common image file extensions
            # image_extensions = ('.JPG', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')
            image_extensions = ('.toml')

            # List to hold the image file names
            image_files = []

            # Loop through all files in the directory
            for filename in os.listdir(folder_path):
                # Check if the file is an image
                if filename.lower().endswith(image_extensions):
                    image_files.append(filename)
            np_image_files = np.array(image_files)
            return np_image_files

print(data.keys())



variable = "Number of matches"
plt.figure(figsize=(10, 5))
plt.plot(data["Number of matches"], marker='o')
plt.title(key)
plt.xlabel('Value')
plt.ylabel(data["Number of matches"])
plt.grid(True)
plt.show()
