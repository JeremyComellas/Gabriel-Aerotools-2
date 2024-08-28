# Code used as an optimization algorithm to find the best parameters for the tracker detection on the thing
# this iteration does an attempt at combining both across images, and across methods. 
# It can be done for any method of stitching (floor, sepparation, surface), you simply have to change the function within the optimizer. So change the stitching function


import os
import cv2
import time
import toml
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

import geopandas as gpd
from pyproj import Transformer
from PIL import Image
from PIL.ExifTags import TAGS



def find_best_parameters(im_num0, im_numf, porcentaje = 0.1, analyse=False, pix_overlap=0, correction=0,  logger=True, plot=False, plotMASK=False, plot_altering=False, plot_matches=False, version="", folder_tag="", folder_save="Data_processed/main/", folder_path_input="Data_processed/Split_ini/__0/",  compare=False, record=False, contar= False, INFO_MATCH=False):
    if logger:
        def print_n_save(*args, **kwargs):
            # Open a file in append mode
            with open(outputfile, 'a') as f:
                print(*args, file=f, **kwargs)
            # Print to terminal
            print(*args, **kwargs)
    else:
        def print_n_save(content):
            print(content)

    # Get the current date and time for text file name. 
    now = datetime.now()
    # Format the current date and time to be file-system-friendly
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the location where the outputs are going to be placed. 
    folder = folder_save
    directory_path = folder + f"Analysis_Images_{im_num0}_{im_numf}_{folder_tag}/"
    os.makedirs(directory_path, exist_ok=True)
    outputfile=  directory_path + f"data_log_{str(timestamp)}.txt"

    # Initialize the document
    print_n_save("Starting the process for saving the data from the terminal into the text file")

    # Start the timer
    start_time = time.time()
    # print_n_save(f"- Timer initialized at {start_time:.2f} s")
    print_n_save(f"- Timer initialized s")
    
    def load_images_inv(image_num, zoom_fact, folder_path1='Data/Fotos_RGB/'):

        def zoom_in(image, zoom_factor=1.2):
            # Get the dimensions of the image
            height, width = image.shape[:2]
            # Calculate the center of the image
            center_x, center_y = width // 2, (height // 2) #- 300
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
            zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)       #cv2.INTER_LINEAR   cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
            return zoomed_image
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


        # image1 = cv2.imread('Data/Fotos_RGB/DJI_20240530104635_0002_W_point1.JPG')
        # image2 = cv2.imread('Data/Fotos_RGB/DJI_20240530104632_0001_W_point1.JPG')


        folder_path_1 = folder_path1
        folder_path_2 = 'Data/Fotos_Termo/'
        image_names_1 = get_picture_filenames(folder_path_1)
        # image_names_2 = get_picture_filenames(folder_path_2)

        cual = image_num    # 479  # 363-396       it moves to the left, so picture i+1 is equivalent to shifting picture i to the right. # Im not that sure about this anymore
        # Comparison between images
        image1 = folder_path_1 + image_names_1[cual]
        image2 = folder_path_1 + image_names_1[cual-1]
        # print_n_save(f"Images read are: \n{image1}\n{image2}")
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image1 = zoom_in(image1, zoom_fact)
        image2 = zoom_in(image2, zoom_fact)

        return image1, image2
    def load_images_orig(image_num, zoom_fact, folder_path1='Data/Fotos_RGB/'):

        def zoom_in(image, zoom_factor=1.2):
            # Get the dimensions of the image
            height, width = image.shape[:2]
            # Calculate the center of the image
            center_x, center_y = width // 2, (height // 2) #- 300
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
            zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)       #cv2.INTER_LINEAR   cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
            return zoomed_image
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


        # image1 = cv2.imread('Data/Fotos_RGB/DJI_20240530104635_0002_W_point1.JPG')
        # image2 = cv2.imread('Data/Fotos_RGB/DJI_20240530104632_0001_W_point1.JPG')


        folder_path_1 = folder_path1
        folder_path_2 = 'Data/Fotos_Termo/'
        image_names_1 = get_picture_filenames(folder_path_1)
        # image_names_2 = get_picture_filenames(folder_path_2)

        cual = image_num    # 479  # 363-396       it moves to the left, so picture i+1 is equivalent to shifting picture i to the right. # Im not that sure about this anymore
        # Comparison between images
        image1 = folder_path_1 + image_names_1[cual]
        image2 = folder_path_1 + image_names_1[cual+1]
        # print_n_save(f"Images read are: \n{image1}\n{image2}")
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image1 = zoom_in(image1, zoom_fact)
        image2 = zoom_in(image2, zoom_fact)

        return image1, image2

    ini  = im_num0
    fini = im_numf

    # This is the optimized version of the obtaining GPS coordiantes function, which takes less time:
    def obtain_GPSINFO_OP(folder_path_originals='Data/Fotos_RGB/'):
        def extract_loc_imag_simpl(folder_path_1, all=False):
            def convert_gps_info(gps_info):
                def dms_to_decimal(degrees, minutes, seconds, direction):
                    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
                    if direction in ['S', 'W']:
                        decimal = -decimal
                    return decimal

                latitude_dms = gps_info[2]
                latitude_dir = gps_info[1]
                longitude_dms = gps_info[4]
                longitude_dir = gps_info[3]
                altitude = gps_info[6]
                
                latitude = dms_to_decimal(latitude_dms[0], latitude_dms[1], latitude_dms[2], latitude_dir)
                longitude = dms_to_decimal(longitude_dms[0], longitude_dms[1], longitude_dms[2], longitude_dir)
                
                return float(latitude), float(longitude), float(altitude)

            def convert_time_info(time_info:str):
                datetime_format = '%Y:%m:%d %H:%M:%S'
                datetime_obj = datetime.strptime(time_info, datetime_format)
                return datetime_obj

            def get_picture_filenames(folder_path):
                image_extensions = ('.JPG', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif')
                image_files = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(image_extensions)]
                return np.array(image_files)
            
            image_names_1 = get_picture_filenames(folder_path_1)

            print_n_save("\nExtracting locations from images")
            im_0, im_F, im_9 = 0, 1, 600
            if all:
                im_9 = len(image_names_1)
            
            GPSInfo1_list = []
            Image_path1_list = []
            Time_pic1_list = []

            for i in range(im_0, im_9, im_F):
                image_path1 = folder_path_1 + image_names_1[i]
                Image_path1_list.append(image_path1)

                with Image.open(image_path1) as image_meta1:
                    exif_data1 = image_meta1._getexif()
                    metadata1 = {}
                    if exif_data1 is not None:
                        for tag_id, value in exif_data1.items():
                            tag = TAGS.get(tag_id, tag_id)
                            metadata1[tag] = value
                    GPSInfo1_list.append(convert_gps_info(metadata1['GPSInfo']))
                    Time_pic1_list.append(convert_time_info(metadata1['DateTimeOriginal']))

                total_amount = im_9 // im_F 
                if i % (im_F * 20) == 0:
                    print_n_save(f"Successfully extracted location from image {i - im_0 // im_F} out of {total_amount - im_0}")

            GPSInfo1 = np.array(GPSInfo1_list)
            Image_path1 = np.array(Image_path1_list)
            Time_pic1 = np.array(Time_pic1_list)

            return GPSInfo1, Image_path1, Time_pic1

        def transform_GPS_proj_simpl(GPSInfo1_p):
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719")
            for row in GPSInfo1_p:
                easting, northing = transformer.transform(row[0], row[1])
                row[0] = float(easting)
                row[1] = float(northing)
            return GPSInfo1_p

        GPSInfo1, Image_path1, Time_pic1 = extract_loc_imag_simpl(folder_path_originals, all=True)
        GPSInfo1_p = GPSInfo1[:,:2]
        GPSInfo1_p = transform_GPS_proj_simpl(GPSInfo1_p)
        
        return GPSInfo1
    GPS = obtain_GPSINFO_OP(folder_path_originals='Data/Fotos_RGB/')



    # Load the images and find masks 
    results = []
    print_n_save(f"\nStart loading and finding masks for images: \n{ini} - {fini}")
    previous_time = time.time()
    #==================================================================================================
    for i in range(ini, fini+1):

        im_num = i
        def check_y_difference(GPSInfo1_p, i):  # To find the mode
            # Ensure the index is within the valid range
            if i < 0 or i >= len(GPSInfo1_p) - 1:
                raise IndexError("Index out of range for array of length {}".format(len(GPSInfo1_p)))

            # Compute the difference in Y values
            y_diff = GPSInfo1_p[i + 1][1] - GPSInfo1_p[i][1]

            # Return the appropriate string based on the difference
            if y_diff > 0:
                return "orig"
            else:
                return "inv"
        mode = check_y_difference(GPS, im_num)    

        if mode == "orig":
            image1, image2 = load_images_orig(im_num-1, 1, folder_path1=folder_path_input)
            image_numb1 = im_num
            image_numb2 = im_num + 1
            # if first>=1:
            #     image1 = joined_image
        elif mode == "inv":
            image1, image2 = load_images_inv(im_num-1, 1, folder_path1=folder_path_input)
            image_numb1 = im_num
            image_numb2 = im_num - 1
            # if first>=1:
            #     image2 = joined_image

        def row_detect(image1, im_numb, threshold= 0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=False, save=False):

            LOR = YOLO(detect_model_path)

            # ASIGNAR INFERENCIA
            detection = LOR(image1)[0]
            boxes = np.empty(4)
            
            # Check for detections in the image
            if len(detection.boxes.data.tolist()) == 0:
                print_n_save("Nothing identified")
            else:
                # Loop through the detections
                for instance in detection.boxes.data.tolist():
                    x1, y1, x2, y2, diode_score, class_id = instance
                    coords = np.array([int(x1), int(y1), int(x2), int(y2)])
                    boxes = np.vstack((boxes, coords))

                    if plot:
                        image_d = image1.copy()
                        # Only process detections above the threshold
                        if diode_score >= threshold:
                            # Draw the bounding box
                            cv2.rectangle(image_d, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 8)

                            # Put the class label and confidence score on the image
                            label = f"Confidence : {diode_score:.2f}"           # VERY USEFUL NOTATION 
                            font_scale = 2  # Increased font scale
                            text_thickness = 3  # Increased thickness
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                            label_x = int(x1)
                            label_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10

                            # Draw label background rectangle
                            cv2.rectangle(image_d, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y + label_size[1]), (0, 255, 0), -1)

                            # Draw label text
                            cv2.putText(image_d, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

            boxes = boxes[1:,:]
            
            if plot:
                plt.figure(figsize=(12, 8))
                plt.title(f'Segmentación de Trackers {im_numb}')
                plt.imshow(cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB))

                if save:
                    plt.savefig(f"Outputeado/Row_detection/Row_detect_{im_numb}")
                else:
                    plt.show()

            return boxes
        def create_mask_TRACKER(image, boxes, porcen=0):
            # Initialize the mask with ones (1s)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Loop through each box and set the corresponding area in the mask to zero (0s)
            for box in boxes:
                x1, y1, x2, y2 = box

                height = y2-y1
                pixels_cut = int(height*porcentaje)
                mask[int(y1+pixels_cut):int(y2-pixels_cut), int(x1):int(x2)] = 1

            return mask
        
        porcentaje = porcentaje

        if mode == "orig":
            im_BEG = image1[:, :-3700]
            im_END = image1[:, -3700:]
            boxes1 = row_detect(im_END, image_numb1, threshold=0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=plot, save=False)
            boxes2 = row_detect(image2, image_numb2, threshold=0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=plot, save=False)

            mask_1_sec = create_mask_TRACKER(im_END, boxes1, porcentaje)
            mask_1 = mask_1_sec
            mask_2 = create_mask_TRACKER(image2, boxes2, porcentaje)
            image1 = im_END

        if mode == "inv":
            im_BEG = image2[:, :3700]
            im_END = image2[:, 3700:]
            boxes1 = row_detect(image1, image_numb1, threshold=0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=plot, save=False)
            boxes2 = row_detect(im_BEG, image_numb2, threshold=0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=plot, save=False)

            mask_2_sec = create_mask_TRACKER(im_BEG, boxes2, porcentaje)
            mask_2 = mask_2_sec
            mask_1 = create_mask_TRACKER(image1, boxes1, porcentaje)
            image2 = im_BEG

        results.append((image1, image2, mask_1, mask_2, image_numb1, image_numb2))
    # ↑ this returns results list
    print_n_save(f"Finished finding masks for all images. Stage took {(time.time()-previous_time):.2f} s")
    previous_time = time.time()



    # =================================================================================================
    # =================================================================================================

    # Iterate over each variable, and create different dictionaries. 
    def vary_parameters(parameter:str, steps:int, plot_matches = plot_matches):
        # global clipLimit, tileGridSize, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, trees, checks, ransacReprojThreshold
        # Creating the nested dictionary
        parameters_dict = {
            'clipLimit': {
                'value': 2.0 ,
                'range': (1.0, 4.0)
            },
            'tileGridSize': {
                'value': (8, 8),
                'range': (4, 16)
            },
            'nfeatures': {
                'value': 0,
                'range': (0, 5000)
            },
            'nOctaveLayers': {
                'value': 3,
                'range': (2, 4)
            },
            'contrastThreshold': {
                'value':  0.04,
                'range': (0.01, 0.1)
            },
            'edgeThreshold': {
                'value': 10,
                'range': (5, 20)
            },
            'sigma': {
                'value': 1.6,
                'range': (1.2, 2.0)
            },
            'trees': {
                'value': 5,
                'range': (1, 16)
            },
            'checks': {
                'value': 50,
                'range': (10, 100)
            },
            'ransacReprojThreshold': {
                'value': 5.0,
                'range': (1.0, 10.0)
            }
        }

        master_dict = {
        "Number of matches": [],
        "Average distance x": [],
        "Average distance y": [],
        "Std_dev distance x": [],
        "Std_dev distance y": [],
        "Percentage good matches over keypoints": [],
        "Average match similitude": [],
        "Number of inliers": [],
        "Inlier ratio": [],
        "Average match size image 1": [],
        "Max match size image 1": [],
        "Min match size image 1": [],
        "Std_dev match size image 1": [],
        "Average match response image 1": [],
        "Std_dev match response image 1": [],
        "Average match size image 2": [],
        "Max match size image 2": [],
        "Min match size image 2": [],
        "Std_dev match size image 2": [],
        "Average match response image 2": [],
        "Std_dev match response image 2": []
        }

        # Default parameters for createCLAHE with reasonable ranges
        clipLimit = 2.0               # Default is 2.0 (threshold for contrast limiting). Range: 1.0 to 4.0
        tileGridSize = (8, 8)         # Default is (8, 8) (size of grid for histogram equalization). Range: (4, 4) to (16, 16)
        # Define parameters for SIFT_create with reasonable ranges
        nfeatures = 0             # The number of best features to retain. If 0, no limit is applied. Range: 0 to several thousand (e.g., 500 to 5000)
        nOctaveLayers = 3         # The number of layers in each octave. 3 is the value used in D. Lowe's paper. Range: 2 to 4
        contrastThreshold = 0.04  # The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. Range: 0.01 to 0.1
        edgeThreshold = 10        # The threshold used to filter out edge-like features. Range: 5 to 20
        sigma = 1.6               # The sigma of the Gaussian applied to the input image at the octave #0. Range: 1.2 to 2.0
        # Default parameters for FLANN-based matcher with reasonable ranges
        # Index parameters
        algorithm = 1  #constat    # Default is 1 (FLANN_INDEX_KDTREE, used for feature matching with SIFT, SURF, etc.). Range: 0 (linear search) to 6 (composite of different algorithms)
        trees = 5                  # Default is 5 (number of trees used in the K-D tree algorithm). Range: 1 to 16 (more trees increase accuracy but also computation time)
        # Search parameters
        checks = 50                # Default is 50 (number of times the tree(s) in the index should be recursively traversed). Range: 10 to 100 (higher values increase accuracy but also computation time)
        # Parameters for findHomography with reasonable ranges
        method = cv2.RANSAC        # Default is cv2.RANSAC (Robust Estimation Algorithm). Alternatives: cv2.RHO, cv2.LMEDS
        ransacReprojThreshold = 5.0  # Default is 5.0 (maximum allowed reprojection error). Range: 1.0 to 10.0 (lower values are more strict, higher values allow more outliers)
    

        # sepparate into the ones that need float steps, and the ones that need int steps
        if parameter=="clipLimit" or parameter=="contrastThreshold" or parameter=="sigma" or parameter=="ransacReprojThreshold":
            step_size = (parameters_dict[parameter]["range"][1]-parameters_dict[parameter]["range"][0])/steps
            # for i in np.arange(parameters_dict[parameter]["range"][0], parameters_dict[parameter]["range"][1], 1): 
            for i in np.arange(parameters_dict[parameter]["range"][0], parameters_dict[parameter]["range"][1], step_size): 
                if parameter=="clipLimit":
                    clipLimit = parameters_dict[parameter]["value"] + i
                elif parameter=="tileGridSize":
                    tileGridSize = parameters_dict[parameter]["value"] + i
                elif parameter=="nfeatures":
                    nfeatures = parameters_dict[parameter]["value"] + i
                elif parameter=="nOctaveLayers":
                    nOctaveLayers = parameters_dict[parameter]["value"] + int(i)
                elif parameter=="contrastThreshold":
                    contrastThreshold = parameters_dict[parameter]["value"] + i
                elif parameter=="edgeThreshold":
                    edgeThreshold = parameters_dict[parameter]["value"] + i
                elif parameter=="sigma":
                    sigma = parameters_dict[parameter]["value"] + i
                elif parameter=="trees":
                    trees = parameters_dict[parameter]["value"] + i
                elif parameter=="checks":
                    checks = parameters_dict[parameter]["value"] + i
                elif parameter=="ransacReprojThreshold":
                    ransacReprojThreshold = parameters_dict[parameter]["value"] + i

                # ====================
                # Perform matching on the loaded image:        (check variation across methods)
                # ====================
                # for row in results:
                image1, image2, mask_1, mask_2, image_numb1, image_numb2= results[0]
                
                Current_image_time = time.time()

                separador = "----------------------------------------------------------------------------"
                print_n_save(f"{separador}")
                print_n_save(f"\n\n\n{separador}")
                print_n_save(f"Analysis images {image_numb1} and {image_numb2}")
                print_n_save(f"{separador}")
                # Process images
                print_n_save(f"\n\n  - Start image pre-processing")
                previous_time = time.time()
                image1_feat = image1.copy()
                image2_feat = image2.copy()
                # Step 2: Convert the image to grayscale
                image1_gray = cv2.cvtColor(image1_feat, cv2.COLOR_BGR2GRAY)
                image2_gray = cv2.cvtColor(image2_feat, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                image1_feat = clahe.apply(image1_gray)
                image2_feat = clahe.apply(image2_gray)
                print_n_save(f"\n      - Finished image pre-processing, stage took {time.time()-previous_time:.2f} s")



                print_n_save(f"\n\n  - Start feature Detection")
                previous_time = time.time()
                sift = cv2.SIFT_create(nfeatures=nfeatures,
                                    nOctaveLayers=nOctaveLayers,
                                    contrastThreshold=contrastThreshold,
                                    edgeThreshold=edgeThreshold,
                                    sigma=sigma)
                # kp1, des1 = sift.detectAndCompute(image1, mask_1)   # before it was None instead of mask_1
                kp1, des1 = sift.detectAndCompute(image1_feat, mask_1)   # before it was None instead of mask_1
                kp2, des2 = sift.detectAndCompute(image2_feat, mask_2)

                if len(kp1) > 1 and len(kp2) > 1:

                    print_n_save(f"\n      There are {len(kp1)} points in image1 and {len(kp2)} in image2")
                    print_n_save(f"\n      - Finished feature detection, stage took {time.time()-previous_time:.2f} s")
                    
                    print_n_save(f"\n\n  - Start feature Matching")
                    previous_time = time.time()
                    # FLANN-based matcher
                    index_params = dict(algorithm=algorithm, trees=trees)
                    search_params = dict(checks=checks)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                    # Extract location of good matches
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


                    

                    # Find the homography matrix and mask using the specified parameters
                    M, mask = cv2.findHomography(src_pts, dst_pts, method, ransacReprojThreshold)   
                    matches_mask = mask.ravel().tolist()
                    # Apply matches_mask to good_matches
                    good_matches_filtered = [m for m, mask in zip(good_matches, matches_mask) if mask]
                    num_good_matches = len(good_matches_filtered)
                    print_n_save(f"\n      There are {num_good_matches} matches")
                    print_n_save(f"\n      - Finished feature Matching, stage took {time.time()-previous_time:.2f} s")

                    if plot_matches:
                        path_image_matches = directory_path  + f'output_{parameter}_{str(timestamp)}/'
                        os.makedirs(path_image_matches, exist_ok=True)
                        if len(good_matches_filtered)>0:
                            print_n_save(f"\n\n  - Start saving image")
                            previous_time = time.time()

                            # # Plot matches using custom function
                            def plot_matches(image1, image2, matches, keypoints1, keypoints2):     # Returns an image
                                # Create a new output image that concatenates the two images together
                                rows1 = image1.shape[0]
                                cols1 = image1.shape[1]
                                rows2 = image2.shape[0]
                                cols2 = image2.shape[1]

                                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
                                out_img[:rows1, :cols1] = image1
                                out_img[:rows2, cols1:] = image2

                                colors = [
                                    (255, 0, 0),  # Blue
                                    (0, 255, 0),  # Green
                                    (0, 0, 255),  # Red
                                    (255, 255, 0), # Cyan
                                    (255, 0, 255), # Magenta
                                    (0, 255, 255), # Yellow
                                    (128, 0, 128), # Purple
                                    (255, 165, 0), # Orange
                                    (0, 128, 0),   # Dark Green
                                    (0, 0, 128)    # Navy
                                ]
                                c = 0
                                i = len(matches)
                                thickness = 3
                                # Draw the keypoints

                                for match in matches[::-1]:      # Gotta switch around the starting point and account 0 indexing

                                    img1_idx = match.queryIdx
                                    img2_idx = match.trainIdx

                                    (x1, y1) = keypoints1[img1_idx].pt
                                    (x2, y2) = keypoints2[img2_idx].pt
                                    # Draw lines between keypoints
                                    if c==len(colors)-1:
                                        c = 0
                                    color = colors[c]

                                    # Draw circles around keypoints
                                    cv2.circle(out_img, (int(x1), int(y1)), int(keypoints1[img1_idx].size/2), color, 10)
                                    cv2.circle(out_img, (int(x2) + cols1, int(y2)), int(keypoints2[img2_idx].size/2), color, 10)
                                    cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), color, thickness+14)

                                    # DRAW THE MATCH NUMBER
                                    text = f"{i}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 5
                                    font_thickness = 5
                                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                                    # Position for the text and box
                                    text_x = int(x1)
                                    text_y = int(y1)
                                    dim = 25
                                    box_coords = ((text_x - dim, text_y - text_size[1] - dim), (text_x + text_size[0] + dim, text_y + dim))
                                    # Draw the rectangle (colored box)
                                    cv2.rectangle(out_img, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                                    # Put the text inside the rectangle
                                    cv2.putText(out_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

                                    i += -1
                                    c += 1
                                return out_img

                            def mostrar(image, title='Image'):
                                    plt.figure(figsize=(12, 7))
                                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                    plt.title(title)
                                    plt.axis('off')
                                    plt.show()
                            img_matches = plot_matches(image1, image2, good_matches_filtered, kp1, kp2)        # num_matches//2
                            if record:
                                cv2.imwrite(path_image_matches + f"Matches_Images_{parameter}_+{i:.2f}_{version}.jpg", img_matches)   
                            else:
                                mostrar(img_matches)
                        
                            print_n_save(f"\n      - Finished saving image, stage took {time.time()-previous_time:.2f} s")
                        else:
                            print_n_save(f"No matches found, could not plot them")



                    # ====================
                    # Compute Metrics:
                    # ====================

                    # Extract the matched keypoints
                    coord_points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches_filtered])  # 2D array containing the coordinates of the the points in image 1 that are matched
                    coord_points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches_filtered])
                    # Calculate the difference in location of pair of pixels
                    difference_coords =  coord_points2 - coord_points1      # Ensure correct direction ### second minus first, check consistency above
                    
                    def compute__stats(difference_coords):
                        # Compute the initial averages
                        average_x = np.mean(difference_coords[:, 0])
                        average_y = np.mean(difference_coords[:, 1])
                        average_x = int(np.floor(-average_x))
                        average_y = int(np.floor(-average_y))
                        std_dev_x = np.std(difference_coords[:, 0]) # low std = values close to the mean
                        std_dev_y = np.std(difference_coords[:, 1]) # high std = values spread out over wider range
                        return average_x, average_y, std_dev_x, std_dev_y
                    average_x, average_y, std_dev_x, std_dev_y = compute__stats(difference_coords)

                    # Ratio good matches to total number of keypoints:
                    ratio_good_matches = num_good_matches / min(len(kp1), len(kp2))
                    # Average distance of good matches
                    avg_match_distance = np.mean([m.distance for m in good_matches])
                    # Inliners in homography matrix
                    num_inliers = np.sum(mask)
                    inlier_ratio = num_inliers / len(good_matches)  


                    # Analisis on the keypoints
                    def get_keypoint_properties(matches, kp1, kp2):

                        sizes1 = []
                        responses1 = []

                        sizes2 = []
                        responses2 = []

                        for match in matches:      # Gotta switch around the starting point and account 0 indexing
                            img1_idx = match.queryIdx
                            img2_idx = match.trainIdx

                            size = kp1[img1_idx].size                # Diameter of the keypoint neighborhood
                            response = kp1[img1_idx].response        # The response (strength) of the keypoint
                            sizes1.append(size)
                            responses1.append(response)

                            size = kp2[img2_idx].size                # Diameter of the keypoint neighborhood
                            response = kp2[img2_idx].response        # The response (strength) of the keypoint
                            sizes2.append(size)
                            responses2.append(response)

                        av_size1 = np.mean(sizes1)
                        max_size1, min_size1 = np.max(sizes1), np.min(sizes1)
                        av_response1 = np.mean(responses1)
                        std_size1 = np.std(sizes1)
                        std_response1 = np.std(responses1)
                        av_size2 = np.mean(sizes2)
                        max_size2, min_size2 = np.max(sizes2), np.min(sizes2)
                        av_response2 = np.mean(responses2)
                        std_size2 = np.std(sizes2)
                        std_response2 = np.std(responses2)

                        global_sizes = sizes1 + sizes2
                        global_responses = responses1 +responses2

                        return av_size1, max_size1, min_size1, av_response1, std_size1, std_response1, av_size2,  max_size2, min_size2, av_response2, std_size2, std_response2
                    av_size1, max_size1, min_size1, av_response1, std_size1, std_response1, av_size2, max_size2, min_size2, av_response2, std_size2, std_response2 = get_keypoint_properties(good_matches_filtered, kp1, kp2)



                    print_n_save(f"\n\n\n  == Metrics:")
                    print_n_save(f"  -  Varying {parameter}, current value is +{i}")
                    print_n_save(f"\n     Number of matches:     {num_good_matches}")
                    print_n_save(f"     Average distance x:    {average_x}")
                    print_n_save(f"     Average distance y:    {average_y}")
                    print_n_save(f"     Std_dev distance x:             {std_dev_x:.2f}")
                    print_n_save(f"     Std_dev distance y:             {std_dev_y:.2f}")

                    print_n_save(f"\n     Percentage good matches/keypoints:    {(ratio_good_matches)*100:.5f}%")
                    print_n_save(f"     Average match similitude:                 {avg_match_distance:.2f}")
                    print_n_save(f"     Number of inliers:                        {num_inliers}")
                    print_n_save(f"     Inlier ratio:                             {inlier_ratio:.2f}")
                    print_n_save(f"        (Inliners are points that fit well with the calculated homography. A high inliner ratio means \n",
                                "        most of the matches fit well with the homography, indicating strong and accurate alignment between images)")


                    print_n_save(f"\n     Average match size image 1:          {av_size1:.2f}")
                    print_n_save(f"     Max match size image 1:              {max_size1:.2f}")
                    print_n_save(f"     Min match size image 1:              {min_size1:.2f}")
                    print_n_save(f"     Std_dev match size image 1:          {std_size1:.2f}")
                    print_n_save(f"     Average match response image 1:      {av_response1:.2f}")
                    print_n_save(f"     Std_dev match response image 1:      {std_response1:.2f}")
                    print_n_save(f"\n     Average match size image 2:         {av_size2:.2f}")
                    print_n_save(f"     Max match size image 2:              {max_size2:.2f}")
                    print_n_save(f"     Min match size image 2:              {min_size2:.2f}")
                    print_n_save(f"     Std_dev match size image 2:          {std_size2:.2f}")
                    print_n_save(f"     Average match response image 2:      {av_response2:.2f}")
                    print_n_save(f"     Std_dev match response image 2:      {std_response2:.2f}")

                    
                    print_n_save(f"\n Finished step of process, image pair took {(time.time()-Current_image_time):.2f} s")
                    
                    # Store information:
                    # List containing the variables in the specified order
                    metrics_list = [
                        num_good_matches,        # Number of matches
                        average_x,               # Average distance x
                        average_y,               # Average distance y
                        std_dev_x,               # Std_dev distance x
                        std_dev_y,               # Std_dev distance y
                        ratio_good_matches * 100, # Percentage good matches/keypoints
                        avg_match_distance,      # Average match similitude
                        num_inliers,             # Number of inliers
                        inlier_ratio,            # Inlier ratio
                        av_size1,                # Average match size image 1
                        max_size1,               # Max match size image 1
                        min_size1,               # Min match size image 1
                        std_size1,               # Std_dev match size image 1
                        av_response1,            # Average match response image 1
                        std_response1,           # Std_dev match response image 1
                        av_size2,                # Average match size image 2
                        max_size2,               # Max match size image 2
                        min_size2,               # Min match size image 2
                        std_size2,               # Std_dev match size image 2
                        av_response2,            # Average match response image 2
                        std_response2            # Std_dev match response image 2
                    ]

                    # Iteratively add an item to each array
                    for key, item in zip(master_dict.keys(), metrics_list):     # Añade una columna a cada entry del diccionario 
                        master_dict[key].append(item)
                else: 
                    metrics_list = [
                        0,  # Number of matches
                        0,  # Average distance x
                        0,  # Average distance y
                        0,  # Std_dev distance x
                        0,  # Std_dev distance y
                        0,  # Percentage good matches/keypoints
                        0,  # Average match similitude
                        0,  # Number of inliers
                        0,  # Inlier ratio
                        0,  # Average match size image 1
                        0,  # Max match size image 1
                        0,  # Min match size image 1
                        0,  # Std_dev match size image 1
                        0,  # Average match response image 1
                        0,  # Std_dev match response image 1
                        0,  # Average match size image 2
                        0,  # Max match size image 2
                        0,  # Min match size image 2
                        0,  # Std_dev match size image 2
                        0,  # Average match response image 2
                        0   # Std_dev match response image 2
                    ]


                    # Iteratively add an item to each array
                    for key, item in zip(master_dict.keys(), metrics_list):     # Añade una columna a cada entry del diccionario 
                        master_dict[key].append(item)
        
        elif parameter=="tileGridSize" or parameter=="nfeatures" or parameter=="nOctaveLayers" or parameter=="edgeThreshold" or parameter=="trees" or parameter=="checks":
            # for i in np.arange(parameters_dict[parameter]["range"][0], parameters_dict[parameter]["range"][1], 1): 
            for i in np.arange(parameters_dict[parameter]["range"][0], parameters_dict[parameter]["range"][1]+1, 1): 
                if parameter=="clipLimit":
                    clipLimit = parameters_dict[parameter]["value"] + i
                elif parameter=="tileGridSize":
                    tileGridSize = (parameters_dict[parameter]["value"][0] + i, parameters_dict[parameter]["value"][1] + i)
                elif parameter=="nfeatures":
                    nfeatures = parameters_dict[parameter]["value"] + i
                elif parameter=="nOctaveLayers":
                    nOctaveLayers = parameters_dict[parameter]["value"] + i
                elif parameter=="contrastThreshold":
                    contrastThreshold = parameters_dict[parameter]["value"] + i
                elif parameter=="edgeThreshold":
                    edgeThreshold = parameters_dict[parameter]["value"] + i
                elif parameter=="sigma":
                    sigma = parameters_dict[parameter]["value"] + i
                elif parameter=="trees":
                    trees = parameters_dict[parameter]["value"] + i
                elif parameter=="checks":
                    checks = parameters_dict[parameter]["value"] + i
                elif parameter=="ransacReprojThreshold":
                    ransacReprojThreshold = parameters_dict[parameter]["value"] + i

                # ====================
                # Perform matching on the loaded image:        (check variation across methods)
                # ====================
                # for row in results:
                image1, image2, mask_1, mask_2, image_numb1, image_numb2= results[0]
                
                Current_image_time = time.time()

                separador = "----------------------------------------------------------------------------"
                print_n_save(f"{separador}")
                print_n_save(f"\n\n\n{separador}")
                print_n_save(f"Analysis images {image_numb1} and {image_numb2}")
                print_n_save(f"{separador}")
                # Process images
                print_n_save(f"\n\n  - Start image pre-processing")
                previous_time = time.time()
                image1_feat = image1.copy()
                image2_feat = image2.copy()
                # Step 2: Convert the image to grayscale
                image1_gray = cv2.cvtColor(image1_feat, cv2.COLOR_BGR2GRAY)
                image2_gray = cv2.cvtColor(image2_feat, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
                image1_feat = clahe.apply(image1_gray)
                image2_feat = clahe.apply(image2_gray)
                print_n_save(f"\n      - Finished image pre-processing, stage took {time.time()-previous_time:.2f} s")



                print_n_save(f"\n\n  - Start feature Detection")
                previous_time = time.time()
                sift = cv2.SIFT_create(nfeatures=nfeatures,
                                    nOctaveLayers=nOctaveLayers,
                                    contrastThreshold=contrastThreshold,
                                    edgeThreshold=edgeThreshold,
                                    sigma=sigma)
                # kp1, des1 = sift.detectAndCompute(image1, mask_1)   # before it was None instead of mask_1
                kp1, des1 = sift.detectAndCompute(image1_feat, mask_1)   # before it was None instead of mask_1
                kp2, des2 = sift.detectAndCompute(image2_feat, mask_2)

                if len(kp1) > 1 and len(kp2) > 1:

                    print_n_save(f"\n      There are {len(kp1)} points in image1 and {len(kp2)} in image2")
                    print_n_save(f"\n      - Finished feature detection, stage took {time.time()-previous_time:.2f} s")
                    
                    print_n_save(f"\n\n  - Start feature Matching")
                    previous_time = time.time()
                    # FLANN-based matcher
                    index_params = dict(algorithm=algorithm, trees=trees)
                    search_params = dict(checks=checks)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                    # Extract location of good matches
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


                    

                    # Find the homography matrix and mask using the specified parameters
                    M, mask = cv2.findHomography(src_pts, dst_pts, method, ransacReprojThreshold)   
                    matches_mask = mask.ravel().tolist()
                    # Apply matches_mask to good_matches
                    good_matches_filtered = [m for m, mask in zip(good_matches, matches_mask) if mask]
                    num_good_matches = len(good_matches_filtered)
                    print_n_save(f"\n      There are {num_good_matches} matches")
                    print_n_save(f"\n      - Finished feature Matching, stage took {time.time()-previous_time:.2f} s")

                    if plot_matches:
                        path_image_matches = directory_path  + f'output_{parameter}_{str(timestamp)}/'
                        os.makedirs(path_image_matches, exist_ok=True)
                        if len(good_matches_filtered)>0:
                            print_n_save(f"\n\n  - Start saving image")
                            previous_time = time.time()

                            # # Plot matches using custom function
                            def plot_matches(image1, image2, matches, keypoints1, keypoints2):     # Returns an image
                                # Create a new output image that concatenates the two images together
                                rows1 = image1.shape[0]
                                cols1 = image1.shape[1]
                                rows2 = image2.shape[0]
                                cols2 = image2.shape[1]

                                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
                                out_img[:rows1, :cols1] = image1
                                out_img[:rows2, cols1:] = image2

                                colors = [
                                    (255, 0, 0),  # Blue
                                    (0, 255, 0),  # Green
                                    (0, 0, 255),  # Red
                                    (255, 255, 0), # Cyan
                                    (255, 0, 255), # Magenta
                                    (0, 255, 255), # Yellow
                                    (128, 0, 128), # Purple
                                    (255, 165, 0), # Orange
                                    (0, 128, 0),   # Dark Green
                                    (0, 0, 128)    # Navy
                                ]
                                c = 0
                                i = len(matches)
                                thickness = 3
                                # Draw the keypoints

                                for match in matches[::-1]:      # Gotta switch around the starting point and account 0 indexing

                                    img1_idx = match.queryIdx
                                    img2_idx = match.trainIdx

                                    (x1, y1) = keypoints1[img1_idx].pt
                                    (x2, y2) = keypoints2[img2_idx].pt
                                    # Draw lines between keypoints
                                    if c==len(colors)-1:
                                        c = 0
                                    color = colors[c]

                                    # Draw circles around keypoints
                                    cv2.circle(out_img, (int(x1), int(y1)), int(keypoints1[img1_idx].size/2), color, 10)
                                    cv2.circle(out_img, (int(x2) + cols1, int(y2)), int(keypoints2[img2_idx].size/2), color, 10)
                                    cv2.line(out_img, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), color, thickness+14)

                                    # DRAW THE MATCH NUMBER
                                    text = f"{i}"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 5
                                    font_thickness = 5
                                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                                    # Position for the text and box
                                    text_x = int(x1)
                                    text_y = int(y1)
                                    dim = 25
                                    box_coords = ((text_x - dim, text_y - text_size[1] - dim), (text_x + text_size[0] + dim, text_y + dim))
                                    # Draw the rectangle (colored box)
                                    cv2.rectangle(out_img, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                                    # Put the text inside the rectangle
                                    cv2.putText(out_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

                                    i += -1
                                    c += 1
                                return out_img

                            def mostrar(image, title='Image'):
                                    plt.figure(figsize=(12, 7))
                                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                    plt.title(title)
                                    plt.axis('off')
                                    plt.show()
                            img_matches = plot_matches(image1, image2, good_matches_filtered, kp1, kp2)        # num_matches//2
                            if record:
                                cv2.imwrite(path_image_matches + f"Matches_Images_{parameter}_+{i:.2f}_{version}.jpg", img_matches)   
                            else:
                                mostrar(img_matches)
                        
                            print_n_save(f"\n      - Finished saving image, stage took {time.time()-previous_time:.2f} s")
                        else:
                            print_n_save(f"No matches found, could not plot them")



                    # ====================
                    # Compute Metrics:
                    # ====================

                    # Extract the matched keypoints
                    coord_points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches_filtered])  # 2D array containing the coordinates of the the points in image 1 that are matched
                    coord_points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches_filtered])
                    # Calculate the difference in location of pair of pixels
                    difference_coords =  coord_points2 - coord_points1      # Ensure correct direction ### second minus first, check consistency above
                    
                    def compute__stats(difference_coords):
                        # Compute the initial averages
                        average_x = np.mean(difference_coords[:, 0])
                        average_y = np.mean(difference_coords[:, 1])
                        average_x = int(np.floor(-average_x))
                        average_y = int(np.floor(-average_y))
                        std_dev_x = np.std(difference_coords[:, 0]) # low std = values close to the mean
                        std_dev_y = np.std(difference_coords[:, 1]) # high std = values spread out over wider range
                        return average_x, average_y, std_dev_x, std_dev_y
                    average_x, average_y, std_dev_x, std_dev_y = compute__stats(difference_coords)

                    # Ratio good matches to total number of keypoints:
                    ratio_good_matches = num_good_matches / min(len(kp1), len(kp2))
                    # Average distance of good matches
                    avg_match_distance = np.mean([m.distance for m in good_matches])
                    # Inliners in homography matrix
                    num_inliers = np.sum(mask)
                    inlier_ratio = num_inliers / len(good_matches)  


                    # Analisis on the keypoints
                    def get_keypoint_properties(matches, kp1, kp2):

                        sizes1 = []
                        responses1 = []

                        sizes2 = []
                        responses2 = []

                        for match in matches:      # Gotta switch around the starting point and account 0 indexing
                            img1_idx = match.queryIdx
                            img2_idx = match.trainIdx

                            size = kp1[img1_idx].size                # Diameter of the keypoint neighborhood
                            response = kp1[img1_idx].response        # The response (strength) of the keypoint
                            sizes1.append(size)
                            responses1.append(response)

                            size = kp2[img2_idx].size                # Diameter of the keypoint neighborhood
                            response = kp2[img2_idx].response        # The response (strength) of the keypoint
                            sizes2.append(size)
                            responses2.append(response)

                        av_size1 = np.mean(sizes1)
                        max_size1, min_size1 = np.max(sizes1), np.min(sizes1)
                        av_response1 = np.mean(responses1)
                        std_size1 = np.std(sizes1)
                        std_response1 = np.std(responses1)
                        av_size2 = np.mean(sizes2)
                        max_size2, min_size2 = np.max(sizes2), np.min(sizes2)
                        av_response2 = np.mean(responses2)
                        std_size2 = np.std(sizes2)
                        std_response2 = np.std(responses2)

                        global_sizes = sizes1 + sizes2
                        global_responses = responses1 +responses2

                        return av_size1, max_size1, min_size1, av_response1, std_size1, std_response1, av_size2,  max_size2, min_size2, av_response2, std_size2, std_response2
                    av_size1, max_size1, min_size1, av_response1, std_size1, std_response1, av_size2, max_size2, min_size2, av_response2, std_size2, std_response2 = get_keypoint_properties(good_matches_filtered, kp1, kp2)



                    print_n_save(f"\n\n\n  == Metrics:")
                    print_n_save(f"  -  Varying {parameter}, current value is +{i}")
                    print_n_save(f"\n     Number of matches:     {num_good_matches}")
                    print_n_save(f"     Average distance x:    {average_x}   (should be positive)")
                    print_n_save(f"     Average distance y:    {average_y}")
                    print_n_save(f"     Std_dev distance x:             {std_dev_x:.2f}")
                    print_n_save(f"     Std_dev distance y:             {std_dev_y:.2f}")

                    print_n_save(f"\n     Percentage good matches/keypoints:    {(ratio_good_matches)*100:.5f}%")
                    print_n_save(f"     Average match similitude:                 {avg_match_distance:.2f}")
                    print_n_save(f"     Number of inliers:                        {num_inliers}")
                    print_n_save(f"     Inlier ratio:                             {inlier_ratio:.2f}")
                    print_n_save(f"        (Inliners are points that fit well with the calculated homography. A high inliner ratio means \n",
                                "        most of the matches fit well with the homography, indicating strong and accurate alignment between images)")


                    print_n_save(f"\n     Average match size image 1:          {av_size1:.2f}")
                    print_n_save(f"     Max match size image 1:              {max_size1:.2f}")
                    print_n_save(f"     Min match size image 1:              {min_size1:.2f}")
                    print_n_save(f"     Std_dev match size image 1:          {std_size1:.2f}")
                    print_n_save(f"     Average match response image 1:      {av_response1:.2f}")
                    print_n_save(f"     Std_dev match response image 1:      {std_response1:.2f}")
                    print_n_save(f"\n     Average match size image 2:         {av_size2:.2f}")
                    print_n_save(f"     Max match size image 2:              {max_size2:.2f}")
                    print_n_save(f"     Min match size image 2:              {min_size2:.2f}")
                    print_n_save(f"     Std_dev match size image 2:          {std_size2:.2f}")
                    print_n_save(f"     Average match response image 2:      {av_response2:.2f}")
                    print_n_save(f"     Std_dev match response image 2:      {std_response2:.2f}")

                    
                    print_n_save(f"\n Finished step of process, image pair took {(time.time()-Current_image_time):.2f} s")
                    
                    # Store information:
                    # List containing the variables in the specified order
                    metrics_list = [
                        num_good_matches,        # Number of matches
                        average_x,               # Average distance x
                        average_y,               # Average distance y
                        std_dev_x,               # Std_dev distance x
                        std_dev_y,               # Std_dev distance y
                        ratio_good_matches * 100, # Percentage good matches/keypoints
                        avg_match_distance,      # Average match similitude
                        num_inliers,             # Number of inliers
                        inlier_ratio,            # Inlier ratio
                        av_size1,                # Average match size image 1
                        max_size1,               # Max match size image 1
                        min_size1,               # Min match size image 1
                        std_size1,               # Std_dev match size image 1
                        av_response1,            # Average match response image 1
                        std_response1,           # Std_dev match response image 1
                        av_size2,                # Average match size image 2
                        max_size2,               # Max match size image 2
                        min_size2,               # Min match size image 2
                        std_size2,               # Std_dev match size image 2
                        av_response2,            # Average match response image 2
                        std_response2            # Std_dev match response image 2
                    ]

                    # Iteratively add an item to each array
                    for key, item in zip(master_dict.keys(), metrics_list):     # Añade una columna a cada entry del diccionario 
                        master_dict[key].append(item)
                else: 
                    metrics_list = [
                        0,  # Number of matches
                        0,  # Average distance x
                        0,  # Average distance y
                        0,  # Std_dev distance x
                        0,  # Std_dev distance y
                        0,  # Percentage good matches/keypoints
                        0,  # Average match similitude
                        0,  # Number of inliers
                        0,  # Inlier ratio
                        0,  # Average match size image 1
                        0,  # Max match size image 1
                        0,  # Min match size image 1
                        0,  # Std_dev match size image 1
                        0,  # Average match response image 1
                        0,  # Std_dev match response image 1
                        0,  # Average match size image 2
                        0,  # Max match size image 2
                        0,  # Min match size image 2
                        0,  # Std_dev match size image 2
                        0,  # Average match response image 2
                        0   # Std_dev match response image 2
                    ]


                    # Iteratively add an item to each array
                    for key, item in zip(master_dict.keys(), metrics_list):     # Añade una columna a cada entry del diccionario 
                        master_dict[key].append(item)
        
        # Dump the dictionary to a TOML string
        toml_string = toml.dumps(master_dict)
        # Write the TOML string to a .toml file
        with open(directory_path + f'output_{parameter}_{str(timestamp)}.toml', 'w') as toml_file:
            toml_file.write(toml_string)
        print_n_save(f"Dictionary successfully dumped to: 'output_{parameter}_{str(timestamp)}.toml'")

        if analyse:
            
            def save_analysis_graph(variable:str, ord:int):
                plt.figure(figsize=(10, 5))
                colora = ((250/255, 169/255, 17/255))
                colorb = ((79/255, 186/255, 30/255))
                plt.plot(master_dict[variable], marker='o', linestyle='-', ms=3, color=colora)
                plt.title(f"Analysis variation {variable}")
                plt.xlabel('Value')
                plt.ylabel(variable)
                plt.grid(True)
                plt.savefig(path_image_matches + f"{str(ord)}_Analysis_{parameter}_{variable}_{version}.jpg")
                plt.close()
            
            save_analysis_graph("Number of matches", 1)
            save_analysis_graph("Average distance x", 2)
            save_analysis_graph("Std_dev distance x", 3)
            save_analysis_graph("Average match similitude", 4)
            save_analysis_graph("Average match size image 1", 5)
            save_analysis_graph("Average match size image 2", 6)
            save_analysis_graph("Average match response image 1", 7)
            save_analysis_graph("Average match response image 2", 8)
            save_analysis_graph("Std_dev match response image 1", 9)
            save_analysis_graph("Std_dev match response image 2", 10)
            save_analysis_graph("Percentage good matches over keypoints", 11)
            save_analysis_graph("Number of inliers", 12)


        return master_dict
    

    vary_parameters("clipLimit", 20)
    vary_parameters("tileGridSize", 20)

    vary_parameters("nOctaveLayers", 20)
    vary_parameters("contrastThreshold", 20)
    vary_parameters("edgeThreshold", 20)
    vary_parameters("sigma", 20)

    vary_parameters("trees", 20)
    vary_parameters("checks", 20)
    vary_parameters("ransacReprojThreshold", 20)


    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    separador = "----------------------------------------------------------------------------"
    print_n_save(f"\n\n\n\n\n{separador}\nFINISHED ENTIRE PROCESS\nElapsed time is: {elapsed_time:.2f} seconds; == {elapsed_time/60:.2f} minutes")
    print_n_save("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    print_n_save("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    print_n_save("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
#Run the function
try:
    find_best_parameters(47, 47, porcentaje=0.2, plot_matches=True, analyse=True, record=True, folder_save="Data_processed/TODO/", folder_tag="")
except Exception as e:
    print("ERROR GARRAFAL; NO HA FUNCIONADO 47")
try:
    find_best_parameters(48, 48, porcentaje=0.2, plot_matches=True, analyse=True, record=True, folder_save="Data_processed/TODO/", folder_tag="")
except Exception as e:
    print("ERROR GARRAFAL; NO HA FUNCIONADO 48")
try:
    find_best_parameters(49, 49, porcentaje=0.2, plot_matches=True, analyse=True, record=True, folder_save="Data_processed/TODO/", folder_tag="")
except Exception as e:
    print("ERROR GARRAFAL; NO HA FUNCIONADO 49")
try:
    find_best_parameters(50, 50, porcentaje=0.2, plot_matches=True, analyse=True, record=True, folder_save="Data_processed/TODO/", folder_tag="")
except Exception as e:
    print("ERROR GARRAFAL; NO HA FUNCIONADO 50")
try:
    find_best_parameters(51, 51, porcentaje=0.2, plot_matches=True, analyse=True, record=True, folder_save="Data_processed/TODO/", folder_tag="")
except Exception as e:
    print("ERROR GARRAFAL; NO HA FUNCIONADO 51")

# _2 means the second iteration using the same set-up, to validate the consistency

# DO A VALIDATOR FOR CONSISTENCY; IN WHICH IT RUNS IT ALL THREE TIMES; AND CHECKS THE CONSISTENCY OF THE REULTS: THEY SHOULD ALL BE EXACTLY THE SAME