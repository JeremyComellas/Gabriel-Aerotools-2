# Final code for Splitting and Saving images depending on the tracker rows. 
#  Structure the image saving approach, to enable efficient retrieval. Include image path choosing

# combine splitting images into a single function, and implement image saving
# Hacer la segmentación basándose en el output de la detección de trackers

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import os


def process_split_images(im_numb, im_path='Data/Fotos_RGB/', folder_NAME="", file_NAME ="", threshold=0.7, plot=True, save=False):

    def load_images_inv_upd(image_num, zoom_fact, im_path='Data/Fotos_RGB/'):

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

        folder_path_1 = im_path
        # folder_path_2 = 'Data/Fotos_Termo/'
        image_names_1 = get_picture_filenames(folder_path_1)
        # image_names_2 = get_picture_filenames(folder_path_2)

        cual = image_num    # 479  # 363-396       it moves to the left, so picture i+1 is equivalent to shifting picture i to the right. # Im not that sure about this anymore
        # Comparison between images
        image1_s = 'Data/Fotos_RGB/' + image_names_1[cual]
        # image2 = 'Data/Fotos_RGB/' + image_names_1[cual-1]
        # print(f"Image read is: \n{image1_s}")#\n{image2}")
        image1 = cv2.imread(image1_s)
        # image2 = cv2.imread(image2)
        image1 = zoom_in(image1, zoom_fact)
        # image2 = zoom_in(image2, zoom_fact)


        return image1, image1_s.split("/")[-1].strip(".JPG") 
    def un_distort(image, corn_pix=300):
        # print(f"image shape {image.shape}")
        height, width = image.shape[:2]
        # Define the source points (corners of the distorted area)
        # These points need to be manually determined based on the image content

        src_points = np.array([
            [corn_pix, 0],  # Top-left corner                    # CHOSEN ARBITRARILY
            [width-corn_pix, 0],  # Top-right corner
            [0, height],  # Bottom-left corner    (x=0, y=3000)
            [width, height]  # Bottom-right corner (x=4000, y=3000)
        ], dtype=np.float32)

        # Define the destination points (where the source points should be mapped to)
        dst_points = np.array([
            [0, 0],  # Top-left corner          (x=0, y=0)
            [width, 0],  # Top-right corner      (x=4000, y=0)
            [0, height],  # Bottom-left corner    (x=0, y=3000)
            [width, height]  # Bottom-right corner (x=4000, y=3000)

        ], dtype=np.float32)
        # Calculate the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation
        output_size = (width, height)
        result = cv2.warpPerspective(image, matrix, output_size)
        return result

    def row_detect(image1, im_numb, threshold=0.7, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=False, save=False):



        LOR = YOLO(detect_model_path)

        # ASIGNAR INFERENCIA
        detection = LOR(image1)[0]
        boxes = np.empty((1,4))
        image1 = image1.copy()
        # Check for detections in the image
        if len(detection.boxes.data.tolist()) == 0:
            print("Nothing identified")
        else:
            # Loop through the detections
            for instance in detection.boxes.data.tolist():
                x1, y1, x2, y2, diode_score, class_id = instance
                coords = np.array([x1, y1, x2, y2])
                boxes = np.vstack((boxes, coords))

                if plot:
                    # Only process detections above the threshold
                    if diode_score >= threshold:
                        # Draw the bounding box
                        cv2.rectangle(image1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 8)

                        # Put the class label and confidence score on the image
                        label = f"Confidence : {diode_score:.2f}"           # VERY USEFUL NOTATION 
                        font_scale = 2  # Increased font scale
                        text_thickness = 3  # Increased thickness
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                        label_x = int(x1)
                        label_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10

                        # Draw label background rectangle
                        cv2.rectangle(image1, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y + label_size[1]), (0, 255, 0), -1)

                        # Draw label text
                        cv2.putText(image1, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

        # print(boxes.shape)
        if boxes.shape[0]>1:
            boxes = boxes[1:,:]


        if plot:
            plt.figure(figsize=(12, 8))
            plt.title(f'Segmentación de Trackers {im_numb}')
            plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

            if save:
                plt.savefig(f"Outputeado/Row_detection/Row_detect_{im_numb}")
            else:
                plt.show()

        return boxes

    def split_image_MAX(image, boxes):

        sorted_indices = np.argsort(boxes[:, 1])[::-1]      # add "[::-1]"  to re-order it such that it starts counting from the bottom-> so does nothing essentially 
        sorted_boxes= boxes[sorted_indices]
        # print("inner boxes", sorted_boxes)
        ini = 0
        fini = image.shape[0]
        segments = []

        if boxes.shape[0] in [0, 1]:
            end = fini
            start = ini
            segment = image[int(start):int(end), :]
            segments.append(segment)
            
        elif boxes.shape[0] == 2:
            end = fini
            start = sorted_boxes[1, 3]
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

            end = sorted_boxes[0, 1]
            start = ini
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

        
        elif boxes.shape[0] == 3:
            end = fini
            start = sorted_boxes[1, 3]
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

            end = sorted_boxes[0, 1]
            start = sorted_boxes[2, 3]
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

            end = sorted_boxes[1, 1]
            start = ini
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)
        else:
            end = fini
            start = sorted_boxes[1, 3]
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

            end = sorted_boxes[0, 1]
            start = sorted_boxes[2, 3]
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)

            end = sorted_boxes[1, 1]
            start = ini
            segment = image[int(start):int(end), :]
            # print("Segmento", segment)
            segments.append(segment)
            print("\n\n\n\n\n\n===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("The amount of detections exceeds 3, only first 3 are taken ")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
            print("===================================================================================")
        return segments

    image1, image1_s = load_images_inv_upd(im_numb-1, 1.65, im_path)
    image1 = un_distort(image1)

    boxes = row_detect(image1, im_numb, threshold=threshold, detect_model_path="AI/Tracker_detection_RGB/best.pt", plot=False, save=False)
    segments = split_image_MAX(image1, boxes)

    if plot:
        # Plot original image
        # plt.figure(figsize=(12, 8))
        # plt.title(f'Original image {image1_s}')
        # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        # plt.show()

        # Save or display the segments
        for idx, segment in enumerate(segments):
            # plt.figure(figsize=(12, 8))
            # plt.title(f'Segment {idx}')
            # plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            if save:
                nombre_f = folder_NAME + f"{image1_s}{file_NAME}_{im_numb}__{idx}.jpg"
                # plt.savefig(nombre_f)
                cv2.imwrite(nombre_f, segment)
            else:
                plt.show()
    return segments

ini = 800
fin = 1100

for i in range(ini, fin+1):
    im_numb = i
    print("\n\n====================================")
    print((i-ini)/(fin-ini)*100)
    print("====================================")
    # image1, image2 = load_images_inv(im_numb-1, 1.65)
    # image1 = un_distort(image1)
    process_split_images(im_numb, im_path='Data/Fotos_RGB/', folder_NAME="Data_processed/Split_ini/", file_NAME ="_SPLT", threshold=0.7, plot=True, save=True)


