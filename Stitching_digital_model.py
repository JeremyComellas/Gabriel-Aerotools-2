# copied from Errors_image_pairs.py
# One big file that does all of the data handling together. 
# Plotting histogram of the errors between locations Histograms of difference in location values between all RGB & Termo images
# Create function to identify which images lie outside of a specified range

# Working on relating the difference in GPS location for each image pair, to the transformation between RGB and Termo image. 
# Not finished 

import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Transformer
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime


def get_picture_filenames(folder_path):
    # List of common image file extensions
    image_extensions = ('.JPG', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')

    # List to hold the image file names
    image_files = []

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.lower().endswith(image_extensions):
            image_files.append(filename)
    np_image_files = np.array(image_files)
    return np_image_files


def convert_gps_info(gps_info):
    def dms_to_decimal(degrees, minutes, seconds, direction):
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if direction in ['S', 'W']:
            decimal = -decimal
        return decimal
    
    # Extract the relevant GPS information
    latitude_dms = gps_info[2]
    latitude_dir = gps_info[1]
    longitude_dms = gps_info[4]
    longitude_dir = gps_info[3]
    altitude = gps_info[6]
    
    # Convert DMS to decimal
    latitude = dms_to_decimal(latitude_dms[0], latitude_dms[1], latitude_dms[2], latitude_dir)
    longitude = dms_to_decimal(longitude_dms[0], longitude_dms[1], longitude_dms[2], longitude_dir)
    
    return float(latitude), float(longitude), float(altitude)  # there  might be a loss of precision by using these floats conversions. Check it later. 

def convert_time_info(time_info:str):
    datetime_format = '%Y:%m:%d %H:%M:%S'
    datetime_obj = datetime.strptime(time_info, datetime_format)
    return datetime_obj

def check_all_time_same(Time_pic1:np.ndarray, Time_pic2:np.ndarray):
    time_dif = Time_pic1 - Time_pic2
    all_same = np.all(time_dif == time_dif[0])
    if not all_same:
        # Find the indexes of elements that are not the same as the reference value
        indexes_not_same = np.where(time_dif != time_dif[0])[0]

        dif_vals = np.empty((0, 1))
        for el in indexes_not_same:
            dif_vals = np.vstack((dif_vals, time_dif[el][0].total_seconds()))

        return all_same, indexes_not_same, dif_vals
    return all_same, None, None


def extract_loc_imag(folder_path_1, image_names_1, folder_path_2, image_names_2, all=False):
    # Extract GPS locations from each image.
    print("Extracting locations from images")
    im_0 = 0        # First image
    im_F = 1       # Image steps
    im_9 = 500      # Last image
    if all:
        im_9 = max(image_names_1.shape)

    GPSInfo1 = np.empty((0, 3))     # Array containing [ -68.82550161  -22.4819525  2490.621     ]
    GPSInfo2 = np.empty((0, 3))     # Array containing [ -68.82550161  -22.4819525  2490.621     ]
    Image_path1 = np.empty((0,1))
    Image_path2 = np.empty((0,1))

    for i in range(im_0, im_9, im_F):   #len(data)
        image_path1 = folder_path_1 + image_names_1[i]
        Image_path1 = np.vstack((Image_path1, image_path1))
        image_path2 = folder_path_2 + image_names_2[i]
        Image_path2 = np.vstack((Image_path2, image_path2))

        image_meta1 = Image.open(image_path1)
        exif_data1 = image_meta1._getexif()
        metadata1 = {}                          # it is possible to add more meta_data as well
        if exif_data1 is not None:
            for tag_id, value in exif_data1.items():
                # Get the tag name instead of tag ID
                tag = TAGS.get(tag_id, tag_id)      # Loop through the EXIF data, converting tag IDs to tag names
                metadata1[tag] = value
        GPSInfo_u1 = np.array(convert_gps_info(metadata1['GPSInfo']))
        GPSInfo1 = np.vstack((GPSInfo1, GPSInfo_u1))


        image_meta2 = Image.open(image_path2)
        exif_data2 = image_meta2._getexif()
        metadata2 = {}                          # it is possible to add more meta_data as well
        if exif_data2 is not None:
            for tag_id, value in exif_data2.items():
                # Get the tag name instead of tag ID
                tag = TAGS.get(tag_id, tag_id)      # Loop through the EXIF data, converting tag IDs to tag names
                metadata2[tag] = value
        GPSInfo_u2 = np.array(convert_gps_info(metadata2['GPSInfo']))
        GPSInfo2 = np.vstack((GPSInfo2, GPSInfo_u2))

        total_amount = im_9//im_F 
        if i % (im_F*200) == 0:
            print(f"Succesfully extracted location from image {i-im_0//im_F} out of {total_amount-im_0}")

    return GPSInfo1, GPSInfo2, Image_path1, Image_path2
def extract_loc_imag2(folder_path_1, image_names_1, folder_path_2, image_names_2, all=False):
    # Extract GPS locations from each image, as well as the timestamp of their creation.
    print("Extracting locations from images")
    im_0 = 0        # First image
    im_F = 1       # Image steps
    im_9 = 600      # Last image
    if all:
        im_9 = max(image_names_1.shape)

    GPSInfo1 = np.empty((0, 3))     # Array containing [ -68.82550161  -22.4819525  2490.621     ]
    GPSInfo2 = np.empty((0, 3))     # Array containing [ -68.82550161  -22.4819525  2490.621     ]
    Image_path1 = np.empty((0,1))
    Image_path2 = np.empty((0,1))
    Time_pic1 = np.empty((0,1))
    Time_pic2 = np.empty((0,1))

    for i in range(im_0, im_9, im_F):   #len(data)
        image_path1 = folder_path_1 + image_names_1[i]
        Image_path1 = np.vstack((Image_path1, image_path1))
        image_path2 = folder_path_2 + image_names_2[i]
        Image_path2 = np.vstack((Image_path2, image_path2))

        image_meta1 = Image.open(image_path1)
        exif_data1 = image_meta1._getexif()
        metadata1 = {}                          # it is possible to add more meta_data as well
        if exif_data1 is not None:
            for tag_id, value in exif_data1.items():
                # Get the tag name instead of tag ID
                tag = TAGS.get(tag_id, tag_id)      # Loop through the EXIF data, converting tag IDs to tag names
                metadata1[tag] = value
        GPSInfo_u1 = np.array(convert_gps_info(metadata1['GPSInfo']))
        GPSInfo1 = np.vstack((GPSInfo1, GPSInfo_u1))
        Time_pic1_u1 = np.array(convert_time_info(metadata1['DateTimeOriginal']))
        Time_pic1 = np.vstack((Time_pic1, Time_pic1_u1))


        image_meta2 = Image.open(image_path2)
        exif_data2 = image_meta2._getexif()
        metadata2 = {}                          # it is possible to add more meta_data as well
        if exif_data2 is not None:
            for tag_id, value in exif_data2.items():
                # Get the tag name instead of tag ID
                tag = TAGS.get(tag_id, tag_id)      # Loop through the EXIF data, converting tag IDs to tag names
                metadata2[tag] = value
        GPSInfo_u2 = np.array(convert_gps_info(metadata2['GPSInfo']))
        GPSInfo2 = np.vstack((GPSInfo2, GPSInfo_u2))
        Time_pic2_u2 = np.array(convert_time_info(metadata2['DateTimeOriginal']))
        Time_pic2 = np.vstack((Time_pic2, Time_pic2_u2))



        total_amount = im_9//im_F 
        if i % (im_F*200) == 0:
            print(f"Succesfully extracted location from image {i-im_0//im_F} out of {total_amount-im_0}")

    return GPSInfo1, GPSInfo2, Image_path1, Image_path2, Time_pic1, Time_pic2


def transform_GPS_proj(GPSInfo1_p, GPSInfo2_p):

    # Define the transformer for WGS84 to UTM Zone 19S
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719")       # Output units are meters [m]
    # Convert from WGS84 to UTM Zone 19S
    for row in GPSInfo1_p:
        easting, northing = transformer.transform(row[0], row[1])
        row[0] = float(easting)    # [m]
        row[1] = float(northing)   # [m]

    for row in GPSInfo2_p:
        easting, northing = transformer.transform(row[0], row[1])
        row[0] = float(easting )   # [m]
        row[1] = float(northing)   # [m]
    return GPSInfo1_p, GPSInfo2_p


def plot_map(ax, shapefile_path, data1_p, data2_p, s_p:int, title="", x_lab="", y_lab="", val= 0.000001):
    # Used to plot a shape and image locations on an axis (MATPLOTLIB) the highlight(s_p) is an integer 
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Calculate the x-axis limits
    x_min, x_max = gdf.total_bounds[0], gdf.total_bounds[2]
    y_min, y_max = gdf.total_bounds[1], gdf.total_bounds[3]

    # Plot the shapefile
    gdf.plot(ax=ax, color='blue', edgecolor='black')

    # Set x-axis limits to plot only the left half
    # val = 0.000001
    x_mid = (x_min + x_max) / 2 * (1 + val* 300)
    y_min *= 1- val 
    y_max *= 1+ val
    ax.set_xlim(x_min, x_mid)
    ax.set_ylim(y_min, y_max)

    # Plot the data points on the same axis
    ax.plot(data2_p[:,0], data2_p[:,1], '^', markersize=4, color='red', mec='k', mew=0.5, label="Thermal Imag")  # No linestyle to ensure only points          
    ax.plot(data1_p[:,0], data1_p[:,1], '.', markersize=7, color='green', mec='k', mew=0.5, label="RGB Imag")  # No linestyle to ensure only points          
    ax.plot(data1_p[s_p,0], data1_p[s_p,1], 'o', markersize=10, color='y', mec='k', mew=1.2)         # Highlight specific point
    ax.plot(data2_p[s_p,0], data2_p[s_p,1], '^', markersize=8, color='y', mec='k', mew=1.2)         # Highlight specific point

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend()

def plot_map3(ax, shapefile_path, shapefile_path2, data1_p, data2_p, s_p:int, title="", x_lab="", y_lab="", val= 0.000001):
    # Used to plot trackers AND Modules
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf2 = gpd.read_file(shapefile_path2)

    # Calculate the x-axis limits
    x_min, x_max = gdf.total_bounds[0], gdf.total_bounds[2]
    y_min, y_max = gdf.total_bounds[1], gdf.total_bounds[3]

    # Plot the shapefile
    gdf.plot(ax=ax, color='blue', edgecolor='black')
    gdf2.plot(ax=ax, color='red', edgecolor='black')

    # Set x-axis limits to plot only the left half
    # val = 0.000001
    x_mid = (x_min + x_max) / 2 * (1 + val* 300)
    y_min *= 1- val 
    y_max *= 1+ val
    ax.set_xlim(x_min, x_mid)
    ax.set_ylim(y_min, y_max)

    # Plot the data points on the same axis
    ax.plot(data2_p[:,0], data2_p[:,1], '^', markersize=4, color='red', mec='k', mew=0.5, label="Thermal Imag")  # No linestyle to ensure only points          
    ax.plot(data1_p[:,0], data1_p[:,1], '.', markersize=7, color='green', mec='k', mew=0.5, label="RGB Imag")  # No linestyle to ensure only points          
    ax.plot(data1_p[s_p,0], data1_p[s_p,1], 'o', markersize=10, color='y', mec='k', mew=1.2)         # Highlight specific point
    ax.plot(data2_p[s_p,0], data2_p[s_p,1], '^', markersize=8, color='y', mec='k', mew=1.2)         # Highlight specific point

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend()

def plot_map2(ax, shapefile_path, data1_p, data2_p, s_p:np.ndarray, title="", x_lab="", y_lab="", val= 0.000001, colorin="y"):
    # Used to plot a shape and image locations on an axis (MATPLOTLIB) highlights(s_p) are an array 
    
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Calculate the x-axis limits
    x_min, x_max = gdf.total_bounds[0], gdf.total_bounds[2]
    y_min, y_max = gdf.total_bounds[1], gdf.total_bounds[3]

    # Plot the shapefile
    gdf.plot(ax=ax, color='blue', edgecolor='black')

    # Set x-axis limits to plot only the left half
    # val = 0.000001
    x_mid = (x_min + x_max) / 2 * (1 + val* 200)
    y_min *= 1- val 
    y_max *= 1+ val
    ax.set_xlim(x_min, x_mid)
    ax.set_ylim(y_min, y_max)

    # Plot the data points on the same axis
    ax.plot(data2_p[:,0], data2_p[:,1], '^', markersize=4, color='red', mec='k', mew=0.5, label="Thermal Imag")  # No linestyle to ensure only points          
    ax.plot(data1_p[:,0], data1_p[:,1], '.', markersize=7, color='green', mec='k', mew=0.5, label="RGB Imag")  # No linestyle to ensure only points          

    for element in s_p:
        # Highlight specific images. 
        ax.plot(data1_p[element,0], data1_p[element,1], 'o', markersize=10, color=colorin, mec='k', mew=1.2)         # Highlight specific point
        ax.plot(data2_p[element,0], data2_p[element,1], '^', markersize=8, color=colorin, mec='k', mew=1.2)         # Highlight specific point

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend()


def plot_image_p(ax, image_path:str, title=""):
    image = cv2.imread(image_path)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.grid(color="g")

def plot_image_a(ax, image:np.ndarray, title=""):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.grid(color="g")


def zoom_in(image, zoom_factor=1.2):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    # Calculate the cropping box
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    left = center_x - new_width // 2
    right = center_x + new_width // 2
    top = center_y - new_height // 2
    bottom = center_y + new_height // 2
    # Crop the image
    cropped_image = image[top:bottom, left:right]
    # Resize the cropped image back to the original size
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)       # cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
    return zoomed_image
def match_images(image_path1, image_path2, alpha = 0.7, ZOOM=1.67):

    def zoom_in(image, zoom_factor=1.2):
        # Get the dimensions of the image
        height, width = image.shape[:2]
        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2
        # Calculate the cropping box
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        left = center_x - new_width // 2
        right = center_x + new_width // 2
        top = center_y - new_height // 2
        bottom = center_y + new_height // 2
        # Crop the image
        cropped_image = image[top:bottom, left:right]
        # Resize the cropped image back to the original size
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)       # cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
        return zoomed_image

    input_RGB = cv2.imread(image_path1)
    input_Termo = cv2.imread(image_path2)
    output_RGB_zoom = zoom_in(input_RGB, ZOOM)

    # Ensure the thermal image is resized to match the zoomed RGB image
    output_Termo_resized = cv2.resize(input_Termo, (output_RGB_zoom.shape[1], output_RGB_zoom.shape[0]))

    # Superimpose images
    superimposed_image = cv2.addWeighted(output_RGB_zoom, alpha, output_Termo_resized, 1 - alpha, 0)

    return output_RGB_zoom, output_Termo_resized, superimposed_image

def plot_BLEND_image(ax, image_path1, image_path2, title=""):
    image = match_images(image_path1, image_path2)
    ax.imshow(cv2.cvtColor(image[2], cv2.COLOR_BGR2RGB))
    ax.set_title(title)


def plot_comparison(im_number, Image_path1, Image_path2, file_path3, data1, data2, show=True, save=False):
    # Paths to Modelo digital
    file_path3 = 'Data/AZABACHE_PB1_VECTORIALES/AZABACHE_PB1_TRACKERS.shp'
    # Create a plot
    title = f"Image {im_number} Comparison"
    file_name = f"Outputs/Comparisons/Image_{im_number}_Comparison"
    # Plotting
    fig, ax = plt.subplots(1, 4, figsize=(14, 6))
    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.85, wspace=0.3)

    image_path1, image_path2 = Image_path1[im_number][0], Image_path2[im_number][0]
    output_RGB_zoom, output_Termo_resized, superimposed_image = match_images(image_path1, image_path2)
    plot_image_a(ax[0], output_RGB_zoom)
    plot_image_a(ax[1], output_Termo_resized)
    plot_image_a(ax[2], superimposed_image)
    # plot_BLEND_image(ax[2], image_path1, image_path2)
    plot_map(ax[3], file_path3, data1, data2, im_number, val=0.0000005)
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()


def create_histogram(loc_dif, bins=30, save=False):
    # Create a histogram
    title = f"Histograms of difference in location values between {max(loc_dif.shape)} RGB & Termo images2"
    file_name = "Outputs/Histogram_difference_location_RGB_Termo"
    bins = bins       # 'auto',  30 
    naranja = (237/255, 162/255, 33/255)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    fig.tight_layout()
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.85, wspace=0.3)

    # First histogram
    ax[0].hist(loc_dif[:, 0], bins, color=naranja, edgecolor='black', alpha=0.7)
    ax[0].set_title("X variation [m] (longitude)")
    ax[0].set_xlabel('X Error [m]')
    ax[0].set_ylabel('Frequency')

    # Second histogram
    ax[1].hist(loc_dif[:, 1], bins, color='g', edgecolor='black', alpha=0.7)
    ax[1].set_title("Y variation [m] (latitude)")
    ax[1].set_xlabel('Y Error [m]')
    ax[1].set_ylabel('Frequency')

    # Combined histogram
    ax[2].hist(loc_dif[:, 0], bins, color=naranja, edgecolor='black', alpha=0.9, label='X Error')
    ax[2].hist(loc_dif[:, 1], bins, color='g', edgecolor='black', alpha=0.7, label='Y Error')
    ax[2].set_title('Histogram of Combined Errors')
    ax[2].set_xlabel('Error [m]')
    ax[2].set_ylabel('Frequency')
    ax[2].legend()

    # Plot 
    if save:
        plt.savefig(file_name)
    plt.show()

def find_loc_data_outside_range(data, range_x=(-0.5, 0), range_y=(-2, 2), decir=False):
    # function to determine which image pairs have a big difference
    # Condition for the first column: values outside -0.5 < x < 0
    condition1 = (data[:, 0] <= range_x[0]) | (data[:, 0] >= range_x[1])

    # Condition for the second column: values outside -2 < y < 2
    condition2 = (data[:, 1] <= range_y[0]) | (data[:, 1] >= range_y[1])

    # Get the indexes
    index_x = np.where(condition1)[0]
    index_y = np.where(condition2)[0]

    # Print the results
    if decir:
        print(f"Indexes of values in the first column outside {range_x[0]} < x < {range_x[1]}:", index_x)
        print(f"Indexes of values in the second column outside {range_y[0]} < y < {range_y[1]}:", index_y)
    return index_x, index_y

def match_images2(im_indx, loc_dif, Image_path1, Image_path2, alpha = 0.7, ZOOM=1.67):

    distance_x = loc_dif[im_indx][0]
    distance_y = loc_dif[im_indx][1]

    # print(f"distance_x: {distance_x} [m]")
    # print(f"distance_y: {distance_y} [m]")

    def convert_distance_pix():
        pass
    
    def zoom_in(image, zoom_factor, dis_x=0, dis_y=0):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        PIX_DIS = 0   # -670 for 186# 478 needs a +70,   

        # Calculate the center of the image
        center_x, center_y = width // 2 + PIX_DIS, height // 2
        # Calculate the cropping box
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        left = center_x - new_width // 2
        right = center_x + new_width // 2
        top = center_y - new_height // 2
        bottom = center_y + new_height // 2
        # Crop the image
        cropped_image = image[top:bottom, left:right]
        # Resize the cropped image back to the original size
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)       # cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
        return zoomed_image

    input_RGB = cv2.imread(Image_path1[im_indx][0])
    print(Image_path1[im_indx][0])
    input_Termo = cv2.imread(Image_path2[im_indx][0])
    print(Image_path2[im_indx][0])
    output_RGB_zoom = zoom_in(input_RGB, ZOOM)

    # Ensure the thermal image is resized to match the zoomed RGB image
    output_Termo_resized = cv2.resize(input_Termo, (output_RGB_zoom.shape[1], output_RGB_zoom.shape[0]))

    # Superimpose images
    superimposed_image = cv2.addWeighted(output_RGB_zoom, alpha, output_Termo_resized, 1 - alpha, 0)

    return input_RGB, output_RGB_zoom, output_Termo_resized, superimposed_image



# Paths to image folders
folder_path_1 = 'Data/Fotos_RGB/'
folder_path_2 = 'Data/Fotos_Termo/'
image_names_1 = get_picture_filenames(folder_path_1)
image_names_2 = get_picture_filenames(folder_path_2)

file_path3 = 'Data/AZABACHE_PB1_VECTORIALES/AZABACHE_PB1_TRACKERS.shp'
file_path4 = 'Data/AZABACHE_PB1_VECTORIALES/AZABACHE_PB1_MODULOS.shp'

# Extract locations from each image.
GPSInfo1, GPSInfo2, Image_path1, Image_path2, Time_pic1, Time_pic2= extract_loc_imag2(folder_path_1, image_names_1, folder_path_2, image_names_2, all=True) 

# Check if all image pairs have the same timestamp. (NOT NEEDED TO RUN CODE)
# all_same, indexes_not_same, dif_vals= check_all_time_same(Time_pic1, Time_pic2)


GPSInfo1_p = GPSInfo1[:,:2]     # only latitude/longitude coordinates for the pictures (ditch altitude)
GPSInfo2_p = GPSInfo2[:,:2]     
GPSInfo1_p, GPSInfo2_p = transform_GPS_proj(GPSInfo1_p, GPSInfo2_p)
data1 = GPSInfo1_p  # Rename variables
data2 = GPSInfo2_p
loc_dif = data1 - data2 

# Create histogram to identify pair difference ranges.
# Create_histogram(loc_dif, bins=50)

# in_x, in_y = find_loc_data_outside_range(loc_dif, range_x=(-0.5, 0), range_y=(-2, 2))       # function 



im_number = 1203 # one less than the image name, due to 0-indexing. 
im_indx = im_number - 1


# print("im_number", im_number)
# print("im_indx", im_indx)
# print("Image_path1 length", len(Image_path1))
# print("Image_path1[im_indx]",Image_path1[im_indx])
# print("\n\n\n\n\n\n")

# input_RGB, output_RGB_zoom, output_Termo_resized, superimposed_image =  match_images2(im_indx, loc_dif, Image_path1, Image_path2)


title = f"Plotting image {im_number} with trackers, and modules"      ########3
fig, ax = plt.subplots( figsize=(14, 6))        # 1, 4,
fig.tight_layout()
fig.suptitle(title, fontsize=16)
fig.subplots_adjust(top=0.85, wspace=0.3)
# # plot_image_a(ax[0], input_RGB)
# plot_image_a(ax[0], output_RGB_zoom)
# plot_image_a(ax[1], output_Termo_resized)
# plot_image_a(ax[2], superimposed_image)
plot_map3(ax, file_path3, file_path4, data1, data2, im_indx) 



title = f"Plotting image {im_number} with ONLY modules"      ########3
fig, ax = plt.subplots( figsize=(14, 6))        # 1, 4,
fig.tight_layout()
fig.suptitle(title, fontsize=16)
fig.subplots_adjust(top=0.85, wspace=0.3)
# # plot_image_a(ax[0], input_RGB)
# plot_image_a(ax[0], output_RGB_zoom)
# plot_image_a(ax[1], output_Termo_resized)
# plot_image_a(ax[2], superimposed_image)
plot_map(ax, file_path4, data1, data2, im_indx) 

plt.show()





"""



# # Create a plot
# title = "Highlighted image pairs that have an error outside of (-2, 2) in y direction (Yellow), \n and image pairs that have a time difference>1 second (white), and the ones common in both (magenta)"       #============================== UPDATE
# save = False
# file_name = "Outputs/Highlighted_image_pairs_error_distance_error_time_and_common"

# # Plotting
# fig, ax = plt.subplots(1, 3, figsize=(14, 6))
# fig.tight_layout()
# fig.suptitle(title, fontsize=16)
# fig.subplots_adjust(top=0.85, wspace=0.3)
# # Plot 
# plot_map2(ax[0], file_path3, data1, data2, in_y)
# plot_map2(ax[1], file_path3, data1, data2, indexes_not_same, colorin='w')
# plot_map2(ax[2], file_path3, data1, data2, common_values, colorin='m')
# if save:
#     plt.savefig(file_name)
# plt.show()






# ============================== Plot 
# # Paths to Modelo digital
# file_path3 = 'Data/AZABACHE_PB1_VECTORIALES/AZABACHE_PB1_TRACKERS.shp'
# # Create a plot
# title = "Difference between location values"
# save = False
# file_name = "Graph1"
# # Plotting
# fig, ax = plt.subplots(1, 3, figsize=(14, 6))
# fig.tight_layout()
# fig.suptitle(title, fontsize=16)
# fig.subplots_adjust(top=0.85, wspace=0.3)

# ax[0].plot(loc_dif[:,0], 'b')
# ax[0].set_title("X variation [m]")
# ax[1].plot(loc_dif[:,1], 'r')
# ax[1].set_title("Y variation [m]")
# # Plot 
# im_number = 50
# plot_map(ax[2], file_path3, data1, data2, im_number)
# if save:
#     plt.savefig(file_name)
# plt.show()




# Common values in both arrays
common_values = np.intersect1d(in_y, indexes_not_same)
# Values present in the first array but not in the second
diff_values = np.setdiff1d(in_y, indexes_not_same)




# print("Start saving...")
# for i in range(20):     # does not do the last value
#     im_number = i
#     plot_comparison(im_number, Image_path1, Image_path2, file_path3, data1, data2, show=False, save=True)
#     print(f"Successfully saved image {im_number}")
"""
