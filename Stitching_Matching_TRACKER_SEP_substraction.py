# Trying to figure out why its not working properly

# This is a copy of Stitching_Matching_floor_MULT_log_opti_4_directions.py, to try and utilize feature detection on the trackers, instead of on the floor.
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




# I THINK THIS CURRENTLY ONLY WORKS FOR ORIG VERSION, CHECK THE PARAMETERS AND LOGIC FOR THE INV VERSION → SECTOR_IM AND WHATNOT. iMAGE2 = JOINED_IMAGE
def stitching_suelo_MULT(im_num0, im_numf, num_matches, pix_overlap, correction=0,  logger=True, plot=False, plotMASK=False, plot_altering=False, plot_matches=False, version="", folder_tag="",folder_save="Data_processed/STITCHES_FLOOR_MULT_ORDERED/", folder_path_input="Data_processed/Split_ini/__0/", detect_model_path="AI/Tracker_detection_RGB/best.pt", compare=False, record=False, contar= False, INFO_MATCH=False):
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

    def mostrar(image, title='Image'):
                        plt.figure(figsize=(12, 7))
                        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        plt.title(title)
                        plt.axis('off')
                        plt.show()
    
    def mostrar2(image1, image2, title='Original Images'):
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(title)
            fig.tight_layout()
            ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            # ax[0].grid('red')
            ax[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            # ax[1].grid('red')
            # plt.savefig("Outputeado/"+title)
            plt.show()
    
    # Get the current date and time for text file name. 
    now = datetime.now()
    # Format the current date and time to be file-system-friendly
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Define the location where the outputs are going to be placed. 
    folder = folder_save

    directory_path = folder + f"Stitched_Images_{im_num0}_{im_numf}_{folder_tag}/"
    os.makedirs(directory_path, exist_ok=True)

    outputfile=  directory_path + f"data_log_{str(timestamp)}.txt"

    # Use the custom print function
    print_n_save("Starting the process for saving the data from the terminal into the text file")

    # Start the timer
    start_time = time.time()
    print_n_save(f"- Timer initialized at {start_time:.2f} s")
    print_n_save(f"\nThe parameters for this run are:")
    print_n_save(f"         Num_matches:        {str(num_matches)}")
    print_n_save(f"         pix_overlap:        {str(pix_overlap)}")
    print_n_save(f"         correction:         {str(correction)}")
    print_n_save(f"         version:            {str(version)}")
    print_n_save(f"         folder_tag:         {str(folder_tag)}")
    print_n_save(f"         folder_save:        {str(folder_save)}")
    print_n_save(f"         folder_path_input:  {str(folder_path_input)}")
    print_n_save(f"         detect_model_path:  {str(detect_model_path)}\n\n")


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
        print_n_save(f"Images read are: \n{image1}\n{image2}")
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
        print_n_save(f"Images read are: \n{image1}\n{image2}")
        image1 = cv2.imread(image1)
        image2 = cv2.imread(image2)
        image1 = zoom_in(image1, zoom_fact)
        image2 = zoom_in(image2, zoom_fact)

        return image1, image2


    ini  = im_num0
    fini = im_numf

    joined_image = 0
    first = 0       # counter for the first time to be different than the rest


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

            print_n_save("Extracting locations from images")
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
                if i % (im_F * 200) == 0:
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

    invalid_matches = 0
    for imeg in range(ini, fini+1):
        print_n_save("==================================================================================================================")
        print_n_save(f"Starting to save image number {imeg} out of {fini} \nWhich is {(imeg-ini)/(fini-ini+0.00001)*100:.2f}%, after {(time.time()-start_time):.2f} seconds from begining")
        print_n_save("==================================================================================================================")
        previous_time = time.time()
        Current_image_time = time.time()

        num_matches = num_matches
        im_num = imeg


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
            print_n_save("\nCurrent mode is Orig\n")
            image1, image2 = load_images_orig(im_num-1, 1, folder_path1=folder_path_input)
            image_numb1 = im_num
            image_numb2 = im_num + 1
            if first>=1:
                image1 = joined_image
        elif mode == "inv":
            print_n_save("\nCurrent mode is Inv\n")
            image1, image2 = load_images_inv(im_num-1, 1, folder_path1=folder_path_input)
            image_numb1 = im_num
            image_numb2 = im_num - 1
            if first>=1:
                image2 = joined_image


        def row_detect(image1, im_numb, threshold= 0.7, detect_model_path=detect_model_path, plot=False, save=False):

            LOR = YOLO(detect_model_path)

            # ASIGNAR INFERENCIA
            detection = LOR(image1)[0]
            boxes = np.empty(4)
            
            # Check for detections in the image
            if len(detection.boxes.data.tolist()) == 0:
                print_n_save("Nothing identified")
            elif len(detection.boxes.data.tolist()) == 1:
                print_n_save("\n\n\n\nOnly one box identified\n\n\n\n")
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
                

            # apply something such that it truly only focuses on the tracker.
            return mask
        porcentaje = 0.1


        print_n_save(f"\n- Start detection of rows, previous stage took {time.time()-previous_time:.2f} s")
        previous_time = time.time()
        if mode == "orig":

            im_BEG = image1[:, :-3800]
            im_END = image1[:, -3800:]
            boxes1 = row_detect(im_END, image_numb1, threshold=0.7, plot=plot, save=False)
            boxes2 = row_detect(image2, image_numb2, threshold=0.7, plot=plot, save=False)

            mask_1_sec = create_mask_TRACKER(im_END, boxes1, porcentaje)
            mask_1 = mask_1_sec
            mask_2 = create_mask_TRACKER(image2, boxes2, porcentaje)
            image1 = im_END

        if mode == "inv":

            im_BEG = image2[:, :3800]
            im_END = image2[:, 3800:]
            boxes1 = row_detect(image1, image_numb1, threshold=0.7, plot=plot, save=False)
            boxes2 = row_detect(im_BEG, image_numb2, threshold=0.7, plot=plot, save=False)

            mask_2_sec = create_mask_TRACKER(im_BEG, boxes2, porcentaje)
            mask_2 = mask_2_sec
            mask_1 = create_mask_TRACKER(image1, boxes1, porcentaje)
            image2 = im_BEG


        if plotMASK:
            def plot_mask(image1, mask1, image2, mask2, im_numb1, im_numb2):
                # Create a copy of the image to overlay the mask
                masked_image = image1.copy()
                # Apply the mask: set masked areas to red (or any color of your choice)
                masked_image[mask1 == 0] = [0, 0, 0]  # Red color for masked areas
                # Plot the original image and the masked image side by side
                plt.figure(figsize=(14, 4))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("Masked Image")
                plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                plt.savefig(directory_path+f"Mask_{im_numb1}-{im_numb2}____{im_numb1}")


                # Create a copy of the image to overlay the mask
                masked_image = image2.copy()
                # Apply the mask: set masked areas to red (or any color of your choice)
                masked_image[mask2 == 0] = [0, 0, 0]  # Red color for masked areas
                # Plot the original image and the masked image side by side
                plt.figure(figsize=(14, 4))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("Masked Image")
                plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                plt.savefig(directory_path+f"Mask_{im_numb1}-{im_numb2}____{im_numb2}")

            print_n_save("Plots of created masks saved")
            plot_mask(image1, mask_1, image2, mask_2, image_numb1, image_numb2)


        def sort_and_calculate_distance_v(boxes):
            if len(boxes) < 2:
                raise ValueError("At least two boxes are required to calculate the distance.")

            # Calculate the middle point for each detection in both x and y coordinates
            middle_points = [( (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ) for box in boxes]

            # Sort boxes based on the middle point's x-coordinate
            sorted_indices = sorted(range(len(middle_points)), key=lambda i: middle_points[i][0])
            sorted_boxes = [boxes[i] for i in sorted_indices]

            # Calculate the horizontal and vertical distances between the first and second detection
            horizontal_distance = int(middle_points[sorted_indices[1]][0] - middle_points[sorted_indices[0]][0])
            vertical_distance = int(middle_points[sorted_indices[1]][1] - middle_points[sorted_indices[0]][1])

            return sorted_boxes, horizontal_distance, vertical_distance

        # Sort boxes and calculate distance
        if len(boxes1) >1:
            sorted_boxes, distance_between_sepx, distance_between_sepy  = sort_and_calculate_distance_v(boxes1)
            print_n_save(f"\n\n\nHorizontal distance between first two detections: {distance_between_sepx} pixels")
            print_n_save(f"Vertical distance between first two detections: {distance_between_sepy} pixels")
        else:
            print_n_save(f"Could not calculate distance between boxes, since there are {len(boxes1)} detected")


        # Pre-Process images to highight differences and features
        print_n_save(f"\n- Start image pre-processing, previous stage took {time.time()-previous_time:.2f} s")

        image1_feat = image1.copy()
        image2_feat = image2.copy()
        # Step 2: Convert the image to grayscale
        image1_gray = cv2.cvtColor(image1_feat, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2_feat, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
        # clipLimit = contrast enhancement, higher values highlight differences more strongly (very high values cause over-amplification of noise)
        # tileGridSize = defines size of grid for which histogram is equalized. smaller ((4,4)) lead to more local contrast enhancement [smaller details more prominent]; and larger grid sizes ((16,16)) more gloval contrast enhancement [smooths out differences]
        image1_feat = clahe.apply(image1_gray)
        image2_feat = clahe.apply(image2_gray)

        if plot_altering:
            mostrar2(image1_gray, image2_gray, "Grayscale images")
            mostrar2(image1_feat, image2_feat, "Enhanced Grayscale images")


        print_n_save(f"\n- Start feature Detection, previous stage took {time.time()-previous_time:.2f} s")
        previous_time = time.time()

        sift = cv2.SIFT_create()
        # kp1, des1 = sift.detectAndCompute(image1, mask_1)   # before it was None instead of mask_1
        kp1, des1 = sift.detectAndCompute(image1_feat, mask_1)   # before it was None instead of mask_1
        kp2, des2 = sift.detectAndCompute(image2_feat, mask_2)
        print_n_save(f"\nThere are {len(kp1)} points in image1 and {len(kp2)} in image2")


        # BACK TO NORMAL IMAGES
        # Step 3: Convert the grayscale image back to a 3-channel image
        # image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        # image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        print_n_save(f"\n- Start feature Matching, previous stage took {time.time()-previous_time:.2f} s")
        previous_time = time.time()

        # FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
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


        print_n_save(f"\n- Start homography creation, previous stage took {time.time()-previous_time:.2f} s")
        previous_time = time.time()
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)      
        matches_mask = mask.ravel().tolist()

        # Apply matches_mask to good_matches
        good_matches_filtered = [m for m, mask in zip(good_matches, matches_mask) if mask]
        print_n_save(f"\nThere are {len(good_matches_filtered)} matches ")

        if plot_matches:
            # # Draw matches
            img_matches = cv2.drawMatches(image1, kp1, image2, kp2, good_matches_filtered, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
            # # Plot matches using custom function
            def plot_matches(image1, image2, matches, num_matches, keypoints1, keypoints2):     # Returns an image
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
                i = num_matches
                thickness = 4
                # Draw the keypoints

                for match in matches[num_matches-1::-1]:      # Gotta switch around the starting point and account 0 indexing

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
            img_matches = plot_matches(image1, image2, good_matches_filtered, num_matches, kp1, kp2)        # num_matches//2
            if record:
                cv2.imwrite(directory_path + f"Matches_Images_{ini}_{imeg}_{version}.jpg", img_matches)   
                # mostrar(img_matches)
            else:
                mostrar(img_matches)


        matches = good_matches_filtered
        keypoints1 = kp1
        keypoints2 = kp2


        # Extract the matched keypoints
        coord_points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])  # 2D array containing the coordinates of the the points in image 1 that are matched
        coord_points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Calculate the difference in location of pair of pixels
        difference_coords =  coord_points2 - coord_points1      # Ensure correct direction ### second minus first, check consistency above

        if INFO_MATCH:

                    # print_n_save(f"Number of matches: {len(matches)}\n")
                    i = 0
                    for match in matches[:num_matches]:

                        # Indices of the matched keypoints
                        query_idx = match.queryIdx          # index descriptor first set
                        train_idx = match.trainIdx          # index descriptor second set

                        # Keypoints
                        kp1 = keypoints1[query_idx].pt
                        kp2 = keypoints2[train_idx].pt

                        # Calculate horizontal and vertical distances   # IMPORTANT
                        horizontal_dist = kp2[0] - kp1[0]       # second minus first, check consistency below
                        vertical_dist = kp2[1] - kp1[1]

                        # Calculate physical distance in pixels
                        physical_distance = np.linalg.norm(np.array(kp1) - np.array(kp2))
                        i += 1
                        # # Uncomment these ones to identify similarity between matches
                        # print_n_save(f"\nMatch with position number {i}")
                        # print_n_save(f"Distance [how similar the descriptors are]: {match.distance}")#, \nQueryIdx (index first set): {query_idx}, TrainIdx (index second set): {train_idx}, ")  # Distance has nothing to do with actual distance, but rather how similar they are.
                        
                        
                        
                        
                        # print_n_save("coordinates of keypoints1 are:  ", kp1)
                        # print_n_save("coordinates of keypoints2 are:   ", kp2)
                        # print_n_save(f"Horizontal Distance (pixels): {horizontal_dist}, "
                            # f"Vertical Distance (pixels): {vertical_dist}, "
                            # f"\nEuclidean Distance (pixels): {physical_distance}")

                    # print_n_save("\n\nFirst list matches coordinate points")
                    # print_n_save(coord_points1[:num_matches,:])
                    # print_n_save("\nSecond list matches coordinate points")
                    # print_n_save(coord_points2[:num_matches,:])
                    print_n_save("\nDifference in x,y coordinates of each match")
                    print_n_save(difference_coords[:num_matches,:])

        # Function to determine start_x and start_y
        def compute_filtered_averages(difference_coords, num_matches, percentage_x=0.30, percentage_y=0.70, contar:bool=False):

            """
            Computes the averages of the x and y coordinates and then recomputes the averages
            using only the values that are within specified percentages of the initial averages.
            X and Y values are paired, so if one X value is rejected, the corresponding Y value is rejected as well, and vice versa.

            Parameters:
            difference_coords (numpy.ndarray): Array of coordinates.
            num_matches (int): Number of matches to consider.
            percentage_x (float): The percentage for filtering the x values around the initial average (default is 30%).
            percentage_y (float): The percentage for filtering the y values around the initial average (default is 30%).

            Returns:
            tuple: The recomputed averages (new_average_x, new_average_y) after filtering.
            """
            # Compute the initial averages
            initial_average_x = np.mean(difference_coords[:num_matches, 0])
            initial_average_y = np.mean(difference_coords[:num_matches, 1])

            # Compute the range for filtering x-coordinates
            lower_bound_x = initial_average_x * (1 - percentage_x)
            upper_bound_x = initial_average_x * (1 + percentage_x)


            # # Compute the range for filtering y-coordinates
            # lower_bound_y = initial_average_y * (1 - percentage_y)          # THIS COMPUTATION NEEDS TO BE IMPROVED
            # upper_bound_y = initial_average_y * (1 + percentage_y)
            # Compute the range for filtering y-coordinates
            lower_bound_y = -2000          # THIS COMPUTATION NEEDS TO BE IMPROVED # CHAPUZA PARA QUE NO DE ERRORES POR AHORA. 
            upper_bound_y = 2000        # wrote these huge values to avoid the problems that arose when dealing with __1 type images, because they had large differences in height
            # He puesto estos pedazo de valores para que por ahora solo los rechaze si realmente son matches mal hechos. 

            

            # Filter the values within the specified percentage of the initial averages
            if initial_average_x >= 0:
                mask_x = (difference_coords[:num_matches, 0] >= lower_bound_x) & (difference_coords[:num_matches, 0] <= upper_bound_x)
            elif initial_average_x < 0:
                mask_x = (difference_coords[:num_matches, 0] <= lower_bound_x) & (difference_coords[:num_matches, 0] >= upper_bound_x)
            
            # if initial_average_y >=0:           #### Same for this, needs to be improved.
            #     mask_y = (difference_coords[:num_matches, 1] >= lower_bound_y) & (difference_coords[:num_matches, 1] <= upper_bound_y)
            # elif initial_average_y <0:
            #     mask_y = (difference_coords[:num_matches, 1] <= lower_bound_y) & (difference_coords[:num_matches, 1] >= upper_bound_y)

            mask_y = (difference_coords[:num_matches, 1] >= lower_bound_y) & (difference_coords[:num_matches, 1] <= upper_bound_y)

            # Combined mask to ensure paired rejection
            combined_mask = mask_x & mask_y

            filtered_values_x = difference_coords[:num_matches, 0][combined_mask]
            filtered_values_y = difference_coords[:num_matches, 1][combined_mask]

            # Compute the new averages using the filtered values
            new_average_x = np.mean(filtered_values_x)
            new_average_y = np.mean(filtered_values_y)

            if contar:
                print_n_save("Initial array X", difference_coords[:num_matches, 0])
                print_n_save(f"Array X == max: {np.max(difference_coords[:num_matches, 0]):.2f}  min: {np.min(difference_coords[:num_matches, 0]):.2f}")
                print_n_save(f"Initial average X: {initial_average_x:.2f}")
                print_n_save(f"bounds for X:  {lower_bound_x:.2f}  and  {upper_bound_x:.2f}, with threshold {percentage_x*100}%")
                print_n_save(f"Final array X {filtered_values_x}")
                print_n_save(f"Final average X:  {new_average_x:.2f} ")
                print_n_save(f"Any values filtered -->  {difference_coords[:num_matches, 0].shape[0] != len(filtered_values_x)}\n\n")

                print_n_save("Initial array Y", difference_coords[:num_matches, 1])
                print_n_save(f"Array Y == max: {np.max(difference_coords[:num_matches, 1]):.2f}  min: {np.min(difference_coords[:num_matches, 1]):.2f}")
                print_n_save(f"Initial average Y: {initial_average_y:.2f}")
                print_n_save(f"bounds for Y:  {lower_bound_y:.2f}  and  {upper_bound_y:.2f}, with threshold {percentage_y*100}%")
                print_n_save(f"Final array Y {filtered_values_y}")
                print_n_save(f"Final average Y: {new_average_y:.2f} ")
                print_n_save(f"Any values filtered -->  {difference_coords[:num_matches, 1].shape[0] != len(filtered_values_y)}\n\n")
                print_n_save(f"New lower bound Y: {lower_bound_y}\nNew upper bound Y: {upper_bound_y}")


            return new_average_x, new_average_y
        
        def compute_filtered_averages_V2(difference_coords, num_matches, percentage_x=0.30, percentage_y=0.70, contar:bool=False):


            # Compute the initial averages
            initial_average_x = np.mean(difference_coords[:num_matches, 0])
            initial_average_y = np.mean(difference_coords[:num_matches, 1])

            # Compute the range for filtering x-coordinates
            lower_bound_x = initial_average_x * (1 - percentage_x)
            upper_bound_x = initial_average_x * (1 + percentage_x)


       
            lower_bound_y = -2000          # THIS COMPUTATION NEEDS TO BE IMPROVED # CHAPUZA PARA QUE NO DE ERRORES POR AHORA. 
            upper_bound_y = 2000        # wrote these huge values to avoid the problems that arose when dealing with __1 type images, because they had large differences in height
            # 

            # Filter the values within the specified percentage of the initial averages
            if initial_average_x >= 0:
                mask_x = (difference_coords[:num_matches, 0] >= lower_bound_x) & (difference_coords[:num_matches, 0] <= upper_bound_x)
            elif initial_average_x < 0:
                mask_x = (difference_coords[:num_matches, 0] <= lower_bound_x) & (difference_coords[:num_matches, 0] >= upper_bound_x)
            
            # if initial_average_y >=0:           #### Same for this, needs to be improved.
            #     mask_y = (difference_coords[:num_matches, 1] >= lower_bound_y) & (difference_coords[:num_matches, 1] <= upper_bound_y)
            # elif initial_average_y <0:
            #     mask_y = (difference_coords[:num_matches, 1] <= lower_bound_y) & (difference_coords[:num_matches, 1] >= upper_bound_y)

            mask_y = (difference_coords[:num_matches, 1] >= lower_bound_y) & (difference_coords[:num_matches, 1] <= upper_bound_y)

            # Combined mask to ensure paired rejection
            combined_mask = mask_x & mask_y

            filtered_values_x = difference_coords[:num_matches, 0][combined_mask]
            filtered_values_y = difference_coords[:num_matches, 1][combined_mask]

            # Compute the new averages using the filtered values
            new_average_x = np.mean(filtered_values_x)
            new_average_y = np.mean(filtered_values_y)

            if contar:
                print_n_save("Initial array X", difference_coords[:num_matches, 0])
                print_n_save(f"Array X == max: {np.max(difference_coords[:num_matches, 0]):.2f}  min: {np.min(difference_coords[:num_matches, 0]):.2f}")
                print_n_save(f"Initial average X: {initial_average_x:.2f}")
                print_n_save(f"bounds for X:  {lower_bound_x:.2f}  and  {upper_bound_x:.2f}, with threshold {percentage_x*100}%")
                print_n_save(f"Final array X {filtered_values_x}")
                print_n_save(f"Final average X:  {new_average_x:.2f} ")
                print_n_save(f"Any values filtered -->  {difference_coords[:num_matches, 0].shape[0] != len(filtered_values_x)}\n\n")

                print_n_save("Initial array Y", difference_coords[:num_matches, 1])
                print_n_save(f"Array Y == max: {np.max(difference_coords[:num_matches, 1]):.2f}  min: {np.min(difference_coords[:num_matches, 1]):.2f}")
                print_n_save(f"Initial average Y: {initial_average_y:.2f}")
                print_n_save(f"bounds for Y:  {lower_bound_y:.2f}  and  {upper_bound_y:.2f}, with threshold {percentage_y*100}%")
                print_n_save(f"Final array Y {filtered_values_y}")
                print_n_save(f"Final average Y: {new_average_y:.2f} ")
                print_n_save(f"Any values filtered -->  {difference_coords[:num_matches, 1].shape[0] != len(filtered_values_y)}\n\n")
                print_n_save(f"New lower bound Y: {lower_bound_y}\nNew upper bound Y: {upper_bound_y}")


            return new_average_x, new_average_y

        def filter_FEOS(data, limits=60, plot=False):

            # Find Median Value, and center data around it
            Q2 = np.percentile(data, 40, axis=0)        # FOR IT TO BE PERFECTLY PLACED AT THE MEDIAN IT SHOULD BE 50. buT I HAVE SET IT AT 40 TO PRIORITIZE BETTER RANKED MATCHES
            data_c = data - Q2

            lower_bound = -20
            upper_bound = 20
            filter_x = (data_c[:, 0] > lower_bound) & (data_c[:, 0] < upper_bound)
            filter_y = (data_c[:, 1] > lower_bound) & (data_c[:, 1] < upper_bound)
            
            combined_filter = filter_x & filter_y

            data_f = data[combined_filter]
            
            def is_not_empty(array):
                # return array.size != 0
                return array.size > 5
            

            not_empty = is_not_empty(data_f)

            did_filt = data.shape[0] != data_f.shape[0]

            if plot:


                # Set up the plot
                plt.figure(figsize=(10, 6))  # Set the figure size

                # Create the scatter plot
                plt.scatter(abs(data_c[:, 0]), abs(data_c[:, 1]), color='blue', edgecolor='k', s=100, alpha=0.75)

                # Set axis limits
                plt.xlim(-limits, limits)
                plt.ylim(-limits, limits)  # Invert the Y-axis

                # Add grid lines
                plt.grid(True, linestyle='--', alpha=0.6)

                # Add labels and title
                plt.xlabel('X Coordinate', fontsize=14)
                plt.ylabel('Y Coordinate', fontsize=14)
                plt.title('Scatter Plot of Data Matches', fontsize=16)

                # Add a background color for the plot area
                plt.gca().set_facecolor('#f2f2f2')

                # Show the plot
                plt.show()

            return data_f, not_empty, did_filt

        print_n_save(f"\n- Start filtering outliers, previous stage took {time.time()-previous_time:.2f} s")
        a, not_empty, did_filt = filter_FEOS(difference_coords)

        print_n_save("Results from filter_FEOS are:")
        print_n_save(f"Are there any good matches?: {str(not_empty)}")
        print_n_save(f"\nDid it filter the data?: {did_filt}")
        print_n_save(f"\nNew list: \n{a}")
        print_n_save(f"\nNew average {np.mean(a, axis=0)}")
        if not not_empty:
            invalid_matches += 1
        
         
        # filt_X = 0.7
        # filt_Y = 0.9

        # print_n_save(f"\n- Start filtering outliers, previous stage took {time.time()-previous_time:.2f} s")
        # previous_time = time.time()
        # average_x, average_y = compute_filtered_averages(difference_coords, num_matches, filt_X, filt_Y, contar=contar)

        average_x = np.mean(a[:,0])
        average_y = np.mean(a[:,1])


        start_x = int(np.floor(-average_x))
        start_y = int(np.floor(-average_y))  # check this sign value, I think it is right

        start_x += correction

        if start_x < 0:
            print_n_save("----------------------------------------------------------------------")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save("----------------------------------------------------------------------")
            print_n_save(f"IMAGE NUMBER {im_num0} gave errors")
            print_n_save("----------------------------------------------------------------------")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
            print_n_save("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print_n_save("----------------------------------------------------------------------")

            print_n_save(f"previous start_x was {start_x} pixels")
            start_x += distance_between_sepx
            print_n_save(f"new start_x is {start_x} pixels")

            print_n_save(f"previous start_y was {start_y} pixels")
            start_y += distance_between_sepy
            print_n_save(f"new start_y is {start_y} pixels")
  


        def blend_images3(image1, image2, start_x, overlap_width, start_y=0, line=False, limit=8):       # image 1 the one on the left. image 2 the one on the right.
            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            # print_n_save(f"Height1: {height1}")
            # print_n_save(f"Width1: {width1}")
            # print_n_save(f"Height2: {height2}")
            # print_n_save(f"Width2: {width2}")
            # print_n_save(f"Initial value of start_y inside blend_function {start_y}")

            # Compute the dimensions of the new image
            new_width = max(width1, width2 + start_x)           # Uncomment this to get a perfectly shaped image.
            new_height = max(height1 + abs(start_y), height2 + abs(start_y))    # ESTO NO SE TOCA

            # Create a new image with the computed dimensions and fill it with zeros (black)
            blended_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            # min_height = min(height1+start_y, height2+start_y)  # changed this to include start_y as well. 
            min_height = min(height1, height2)  # changed this to include start_y as well. 
            if min_height==height1:
                print("MINHEIGHT IS IMAGE 1")
            elif min_height==height2:
                print("MINHEIGHT IS IMAGE 2")

            print_n_save(f"Start_x is: {start_x}, and it should be positive")
            # print_n_save("min_height", min_height)
            if abs(height1-height2)> 100:
                # raise ValueError("Warning gabo, there is a large mismatch between the height of image1 and image2, which may cause problems in the vertical alignment. Check the blending function")
                print_n_save("\n=====================================================================================================\n=====================================================================================================\n=====================================================================================================")
                print_n_save("\n=====================================================================================================\n=====================================================================================================\n=====================================================================================================")
                print_n_save("\n=====================================================================================================\n=====================================================================================================\n=====================================================================================================")
                print_n_save("\n=====================================================================================================\n=====================================================================================================\n=====================================================================================================")
                print_n_save("\n=====================================================================================================\n=====================================================================================================\n=====================================================================================================")
                print_n_save("Warning Gabo, there is a large mismatch between the height of image1 and image2, which may cause problems in the vertical alignment. Check the blending function")
            # if abs(start_y)<=limit:
            #     start_y = 0
            if start_y >= 0:
                # Place image1 in the new image
                blended_image[0:height1, 0:width1] = image1
                # print_n_save("TUTUTUTUTUTUTUTUTUTU blended_image", blended_image.shape)
                # print_n_save("TUTUTUTUTUTUTUTUTUTU image1", image1.shape)
                # print_n_save("TUTUTUTUTUTUTUTUTUTU start_x2", start_x)
                # print_n_save("TUTUTUTUTUTUTUTUTUTU image2", image2.shape)
                # Blend the overlapping region    # from start_x to the right
                print_n_save("POSITIVE Y")
                print_n_save("start_y", start_y)
                print("height1", height1)
                print("height2", height2)
                print("height_blended", new_height)
                print("min_height", min_height)
                print("height sliced== min_height", min_height)
                for i in range(overlap_width):
                    alpha = i / overlap_width       # calculate alpha value, alpha varies linearly from 0 to 1 as i goes from 0 to overlap_width - 1.
                    # blended_image[start_y:start_y+min_height, start_x + i] = cv2.addWeighted(image1[start_y:start_y+min_height, start_x + i], 1 - alpha, image2[0:min_height, i], alpha, 0) # this should work
                    blended_image[start_y:min_height, start_x + i] = cv2.addWeighted(image1[start_y:min_height, start_x + i], 1 - alpha, image2[0:min_height-start_y, i], alpha, 0) # this should work
                    # starts at start_x and moves to the right the amount of pixels done
                # Place the remaining part of image2
                # print_n_save("TUTUTUTUTUTUTUTUTUTU", blended_image.shape)
                blended_image[start_y:start_y+height2, start_x + overlap_width:start_x + width2] = image2[:height2, overlap_width:]    #  target shape (1738,0,3) has zero width


            elif start_y < 0:           # POR AHORA ESTOY TRABAJANDO EN LA VERSIÓN NEGATIVA, YA HARÉ LA POSITIVA. --> Ya está hecha la positiva también.
                start_y = -start_y
                # print_n_save(f"Sign_changed start_y is {start_y}")
                blended_image[start_y:start_y+height1, 0:width1] = image1
                # print_n_save("LELELELELELELELE blended image", blended_image.shape)
                # print_n_save("LELELELELELELELE start_x2", start_x)
                print_n_save("NEGATIVE Y")
                print_n_save("start_y", start_y)
                print("height1", height1)
                print("height2", height2)
                print("height_blended", new_height)
                print("height sliced", min_height + start_y)
                # Blend the overlapping region    # from start_x to the right
                for i in range(overlap_width):
                    alpha = i / overlap_width       # calculate alpha value, alpha varies linearly from 0 to 1 as i goes from 0 to overlap_width - 1.
                    # blended_image[start_y:start_y+min_height, start_x + i] = cv2.addWeighted(image1[start_y:, start_x + i], 1 - alpha, image2[start_y:start_y+min_height, i], alpha, 0)
                    blended_image[start_y:min_height, start_x + i] = cv2.addWeighted(image1[0:min_height-start_y, start_x + i], 1 - alpha, image2[start_y:min_height, i], alpha, 0)

                # Place the remaining part of image2
                # blended_image[0:min_height, start_x + overlap_width:start_x + width2] = image2[:min_height, overlap_width:] # this could be height2. 
                blended_image[0:height2, start_x + overlap_width:start_x + width2] = image2[:height2, overlap_width:] # this could be height2. 


            if line:
                # Draw a vertical line at the start_x location
                cv2.line(blended_image, (start_x, 0), (start_x, new_height), (0, 255, 0), 3)

            return blended_image
        

        print_n_save(f"\n- Start blending images, previous stage took {time.time()-previous_time:.2f} s")
        print_n_save(f"Value of start_x is {start_x}")
        previous_time = time.time()
        # joined_image = blend_images(image1, image2, start_x, 10, start_y, line=True, limit=8)
        joined_image = blend_images3(image1, image2, start_x, pix_overlap, start_y, line=False, limit=8)


        def rejuntar_orig(image1, image2, im_numb, start_y=0):
            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            # print_n_save(f"Height1: {height1}")
            # print_n_save(f"Width1: {width1}")
            # print_n_save(f"Height2: {height2}")
            # print_n_save(f"Width2: {width2}")

            INCR = 10
            BASE = 6

            label = str(im_numb)
            # Get the label and its size

            font_scale = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, thickness=2)[0]

            # Position the label at the center of the rectangle
            text_x =  180
            text_y = height1 - int(height1/5)

            # Draw the label on the image
            cv2.putText(image2, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=BASE+INCR, lineType=cv2.LINE_AA)
            cv2.putText(image2, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=BASE, lineType=cv2.LINE_AA)

            # Compute the dimensions of the new image
            new_width = width1 + width2           # Uncomment this to get a perfectly shaped image.
            new_height =  max(height1 + abs(start_y), height2 + abs(start_y))

            # Create a new image with the computed dimensions and fill it with zeros (black)
            rejuntated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            if start_y < 0:
                start_y = -start_y
                rejuntated_image[start_y:start_y+height1, 0:width1] = image1
                rejuntated_image[0:height2, width1:] = image2
            else:
                rejuntated_image[0:height1, 0:width1] = image1
                rejuntated_image[0:height2, width1:] = image2
            return rejuntated_image
        
        def rejuntar_inv(image1, image2, im_numb, start_y=0):
            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            # print_n_save(f"Height1: {height1}")
            # print_n_save(f"Width1: {width1}")
            # print_n_save(f"Height2: {height2}")
            # print_n_save(f"Width2: {width2}")

            INCR = 10
            BASE = 6

            label = str(im_numb)
            # Get the label and its size

            font_scale = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, thickness=2)[0]

            # Position the label at the center of the rectangle
            text_x =  180
            text_y = height1 - int(height1/5)

            # Draw the label on the image
            cv2.putText(image1, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness=BASE+INCR, lineType=cv2.LINE_AA)
            cv2.putText(image1, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness=BASE, lineType=cv2.LINE_AA)

            # Compute the dimensions of the new image
            new_width = width1 + width2           # Uncomment this to get a perfectly shaped image.
            new_height =  max(height1 + abs(start_y), height2 + abs(start_y))

            # Create a new image with the computed dimensions and fill it with zeros (black)
            rejuntated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            if start_y < 0:
                rejuntated_image[0:height1, 0:width1] = image1
                rejuntated_image[0:height2, width1:] = image2
            else:
                rejuntated_image[0:height1, 0:width1] = image1
                rejuntated_image[start_y:start_y+height2, width1:] = image2
            return rejuntated_image

        if mode == "orig":
            joined_image = rejuntar_orig(im_BEG, joined_image, image_numb1, start_y)

        elif mode == "inv":
            joined_image = rejuntar_inv(joined_image, im_END, image_numb1, start_y)   
        
        def crop_black_edges(image, threshold=0.9):
            """
            Crops the top or bottom edge of an image if more than 90% of the pixels in the horizontal direction are black.
            
            Parameters:
            image_path (str): The path to the image to be cropped.
            threshold (float): The proportion of black pixels required to trigger cropping.
            
            Returns:
            np.ndarray: The cropped image.
            """

            height, width = image.shape[:2]
            
            # Function to check if a row is mostly black
            def is_mostly_black(row):
                black_pixel_count = np.sum(row == 0)
                return black_pixel_count / width > threshold
            
            # Check the top edge
            top_crop = 0
            for i in range(height):
                if is_mostly_black(image[i]):
                    top_crop = i + 1
                else:
                    break
            
            # Check the bottom edge
            bottom_crop = height
            for i in range(height-1, -1, -1):
                if is_mostly_black(image[i]):
                    bottom_crop = i
                else:
                    break
            
            # Determine which edge to crop
            if top_crop > 0 and bottom_crop < height:
                if top_crop >= (height - bottom_crop):
                    cropped_image = image[top_crop:, :]
                else:
                    cropped_image = image[:bottom_crop, :]
            elif top_crop > 0:
                cropped_image = image[top_crop:, :]
            elif bottom_crop < height:
                cropped_image = image[:bottom_crop, :]
            else:
                cropped_image = image
            
            return cropped_image

        def mostrar(image, title='Image'):
            plt.figure(figsize=(12, 7))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()

        # # Without cropping 
        joined_image = crop_black_edges(joined_image)
        # mostrar(joined_image)


        if plot:

            def mostrar2(image1, image2, title='Original Images'):
                    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
                    fig.suptitle(title)
                    fig.tight_layout()
                    ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                    # ax[0].grid('red')
                    ax[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
                    # ax[1].grid('red')
                    # plt.savefig("Outputeado/"+title)
                    plt.show()
        
            mostrar2(image1, image2)
            mostrar(joined_image)

        if compare:

            def mostrar_comp(image1, image2, im_comb, image_numb1, image_numb2, start_x, mode, record=False, sav_title=directory_path):

                    import matplotlib.pyplot as plt
                    import matplotlib.gridspec as gridspec

                    # Convert images from BGR (OpenCV format) to RGB (Matplotlib format)
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                    im_comb = cv2.cvtColor(im_comb, cv2.COLOR_BGR2RGB)


                    # Create a figure
                    fig = plt.figure(figsize=(10, 8))

                    # Define the grid layout
                    gs = gridspec.GridSpec(3, 2, height_ratios=[0.2, 2, 2])

                    # Add the title (use a single cell spanning both columns)
                    ax0 = fig.add_subplot(gs[0, :])
                    ax0.text(0.5, 0.5, f'Image Comparison ({mode})', horizontalalignment='center', verticalalignment='center', fontsize=14)
                    ax0.axis('off')

                    # Add image 1
                    ax1 = fig.add_subplot(gs[1, 0])
                    ax1.imshow(image1)
                    ax1.set_title(f'Image {image_numb1}', pad=4, fontsize=12)  
                    # ax1.axis('off')

                    # Add image 2
                    ax2 = fig.add_subplot(gs[1, 1])
                    ax2.imshow(image2)
                    ax2.set_title(f'Image {image_numb2}', pad=4, fontsize=12)
                    # ax2.axis('off')

                    # Add image 3 (use a single cell spanning both columns)
                    ax3 = fig.add_subplot(gs[2, :])
                    ax3.imshow(im_comb)
                    ax3.set_title(f'Stitched images with start_x: {start_x}', pad=4, fontsize=12)
                    # ax3.axis('off')
                    # Create a new set of ticks with increased density
                    num_labels = 20  # Change this to your desired number of labels
                    steps = 200
                    new_ticks = np.linspace(0, im_comb.shape[1]-1, num_labels)
                    new_ticks = np.arange(0, im_comb.shape[1], steps)
                    # Set the new ticks on the horizontal axis
                    ax3.set_xticks(new_ticks)
                    # Draw minor ticks as small ticks
                    ax3.tick_params(axis='x', which='minor', length=4)


                    # Adjust layout
                    plt.tight_layout()
                    if record:
                        plt.savefig(sav_title+mode+f"__{ini}_{imeg}")
                    else:
                        plt.show()
                    plt.close()

            mostrar_comp(image1, image2, joined_image, image_numb1, image_numb2, start_x, mode, record=record)
        
        print_n_save(f"\n- Start saving the images, previous stage took {time.time()-previous_time:.2f} s")
        previous_time = time.time()        
        if record:
            cv2.imwrite(directory_path + f"Images_{ini}_{imeg}_{version}.jpg", joined_image)    
            print_n_save(f"\n- Finished saving the image, of shape {joined_image.shape}\nprevious stage took {time.time()-previous_time:.2f} s\n=====\n=====\nThis image took {time.time()-Current_image_time:.2f} s\n\n\n\n")
        print_n_save(f"For now, there were a total of {invalid_matches} of invalid matches")
        previous_time = time.time()    
        first += 1


    # End the timer
    end_time = time.time()


    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print_n_save(f"FINISHED!\nElapsed time is: {elapsed_time:.2f} seconds; == {elapsed_time/60:.2f} minutes")
    print_n_save(f"There were a total of {invalid_matches} of invalid matches")
    print_n_save("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    print_n_save("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    print_n_save("/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    print_n_save("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
    return joined_image




comienza = 646
termina  = 669

comienza = 65
termina  = 77


PEGADITA1 = stitching_suelo_MULT(comienza, termina, 100, pix_overlap=50, correction=0, version="", folder_tag="V1", plot=False, plotMASK=True, plot_altering=False, plot_matches=True, compare=False, logger=True, record=True, folder_save="Data_processed/STITCHES_TRACKER_SEP_MULTIPLE/", folder_path_input="Data_processed/Split_ini/__0/", detect_model_path="AI/Separacion_trackers/best.pt", INFO_MATCH=True)


# for I in range(comienza, termina):
#     try:
#         PEGADITA1 = stitching_suelo_MULT(I, I, 100, pix_overlap=2, correction=0, version="", folder_tag="", plot=False, plotMASK=True, plot_altering=False, plot_matches=True, compare=False, logger=True, record=True, folder_save="Data_processed/STITCHES_TRACKER_SEP_pairs_subtraction3.1_orig/", folder_path_input="Data_processed/Split_ini/__0/", detect_model_path="AI/Separacion_trackers/best.pt", INFO_MATCH=True)
#     except:
#         pass



# Hacer un algoritmo que sepa por cual va, y si falla, seguir con la siguiente imagen. -> para cuando esté terminado el código y a veces de errores y a veces no (debido a la consistencia del feature detection)
# hacer también que si hay demasiados pocos matches, prueba a hacer el stitching otra vez. 




# Incluir el fix en start_y también. 
# Incluir el fix para cuando solo hay una detección en la imagen de la izquierda. 

