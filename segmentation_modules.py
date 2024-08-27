# Module segmentation



import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import os



def mostrar(image, title='Image'):
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_images_inv(image_num, zoom_fact):

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
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_CUBIC)       #cv2.INTER_LINEAR   cv2.INTER_CUBIC    cv2.INTER_LANCZOS4
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


    image1 = cv2.imread('Data/Fotos_RGB/DJI_20240530104635_0002_W_point1.JPG')        
    image2 = cv2.imread('Data/Fotos_RGB/DJI_20240530104632_0001_W_point1.JPG')        


    folder_path_1 = 'Data/Fotos_RGB/'
    folder_path_2 = 'Data/Fotos_Termo/'
    image_names_1 = get_picture_filenames(folder_path_1)
    image_names_2 = get_picture_filenames(folder_path_2)

    cual = image_num    # 479  # 363-396       it moves to the left, so picture i+1 is equivalent to shifting picture i to the right. # Im not that sure about this anymore
    # Comparison between images
    image1 = 'Data/Fotos_RGB/' + image_names_1[cual]
    image2 = 'Data/Fotos_RGB/' + image_names_1[cual-1]
    print(f"Images read are: \n{image1}\n{image2}")
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    image1 = zoom_in(image1, zoom_fact)
    image2 = zoom_in(image2, zoom_fact)


    return image1, image2
def crop_top(image, crop_height=500):
    """
    Crops the top part of the image by the specified crop_height.

    Parameters:
    - image: The input image as a numpy array.
    - crop_height: The number of pixels to crop from the top of the image.

    Returns:
    - Cropped image.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Check if the crop_height is valid
    if crop_height > height:
        raise ValueError("Crop height is larger than the image height")

    # Crop the image
    cropped_image = image[crop_height:height, 0:width]

    return cropped_image
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


def max_min_cont_y(contour:np.ndarray):
    max = np.max(contour[:,1])
    min = np.min(contour[:,1])
    return max, min

image1, image2 = load_images_inv(9, 1.65)      # 46


image1 = un_distort(image1)
image2 = un_distort(image2)


# model_path = "AI/Tracker_segmentation_RGB/best.pt"
model_path = "AI/Module_segmentation_RGB/best_halfway.pt"


def tracker_segment_name(image1, model_path, verbose=False, plot=False):

    def segment_trackers(image1, tracker_model_path="AI/Tracker_segmentation_RGB/best.pt", verbose=False):

        from ultralytics import YOLO


        # Cargar modelo YOLO para segmentación de trackers
        tracker_model_path = tracker_model_path
        tracker = YOLO(tracker_model_path)

        # Realizar la segmentación de trackers
        tracker_results = tracker.predict(source=image1, task='segment')
        # print("\n\nTracker results length: ", len(tracker_results))

        # Crear una copia de la imagen para dibujar los contornos
        image_with_contours = image1.copy()


        for result in tracker_results:
            if result.masks is not None:
                # Convert masks to numpy array and move to CPU
                masks = result.masks.data.cpu().numpy()

                # Get confidence scores for each box if they exist, otherwise fill with None
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else [None] * len(masks)
                if verbose:
                    print(f"There are {masks.shape[0]} Masks of shape {masks.shape[1:]}:")
        

                contours_lst = []
                # Iterate through each mask
                for i, mask in enumerate(masks):
                    # Get the confidence score for the current mask
                    confidence = confidences[i] if i < len(confidences) else None
                    
                    mask = cv2.resize(mask, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Convert mask to uint8 format and scale to [0, 255]
                    mask = (mask * 255).astype(np.uint8)

                    # Extract contours from the mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # this layer is basically useless. 

                    # print("CONTOURSSSS_______________________________")                                 # per mask, there is ONE contours, which may contain multiple instances of contour.
                    # print(type(contours))
                    # print(len(contours))


                    # # Iterate through each contour
                    for contour in contours:
                        if verbose:
                            print(f"Contour number {i}")
                            # print(type(contour))
                            print((contour.shape))

                        contour = contour.reshape(-1, 2)                  #  The reshape method in numpy changes the shape of an array. The -1 in the reshape method is a special value that means "infer this dimension". When you use -1, numpy calculates the size of this dimension so that the total number of elements matches the original array. In reshape(-1, 2):
                                                                            # -1 tells numpy to automatically determine the size of this dimension based on the total number of elements.
                                                                            # 2 specifies that the second dimension should have a size of 2.
                        # # Calculate the moments of the contour to find its center
                        # M = cv2.moments(contour)
                        # if M["m00"] != 0:
                        #     cX = int(M["m10"] / M["m00"])
                        #     cY = int(M["m01"] / M["m00"])
                        # else:
                        #     cX, cY = 0, 0
                        # Draw the contour on the image
                        contours_lst.append(contour)
                        cv2.drawContours(image_with_contours, [contour], -1, (33, 247, 5), 12)
        
        return image_with_contours, mask, contours_lst

    def find_centre_mid(contours:list):

        n_contours = len(contours)
        contours_centre = np.empty((2))

        for i in range(n_contours): 
            contour = contours[i]
            min_x, max_x = np.min(contour[:, 0]), np.max(contour[:, 0])
            min_y, max_y = np.min(contour[:, 1]), np.max(contour[:, 1])
            mid_x = (max_x + min_x) / 2
            mid_y = (max_y + min_y) / 2
            # print("SOFIA", contour.shape)
            vals = np.array([mid_x, mid_y])
            # print("JJJ", vals.shape)
            contours_centre = np.vstack((contours_centre, vals))
        contours_centre = contours_centre[1:,:]
        return contours_centre
    def sort_points(points):
        # Convert the points to a list of tuples for sorting
        points_list = [tuple(point) for point in points]
        
        # Sort the points first by x coordinate, then by y coordinate
        sorted_points = sorted(points_list, key=lambda point: (point[1], point[0]))
        
        # Convert back to a numpy array
        sorted_points_array = np.array(sorted_points)
        
        return sorted_points_array


    def plot_centers(contours, image):

        def find_centre_mid(contours:list):

            n_contours = len(contours)
            contours_centre = np.empty((2))

            for i in range(n_contours): 
                contour = contours[i]
                min_x, max_x = np.min(contour[:, 0]), np.max(contour[:, 0])
                min_y, max_y = np.min(contour[:, 1]), np.max(contour[:, 1])
                mid_x = (max_x + min_x) / 2
                mid_y = (max_y + min_y) / 2
                # print("SOFIA", contour.shape)
                vals = np.array([mid_x, mid_y])
                # print("JJJ", vals.shape)
                contours_centre = np.vstack((contours_centre, vals))
            contours_centre = contours_centre[1:,:]
            return contours_centre
        def sort_points(points):
            # Convert the points to a list of tuples for sorting
            points_list = [tuple(point) for point in points]
            
            # Sort the points first by x coordinate, then by y coordinate
            sorted_points = sorted(points_list, key=lambda point: (point[1], point[0]))
            
            # Convert back to a numpy array
            sorted_points_array = np.array(sorted_points)
            
            return sorted_points_array
        
        n_contours = len(contours)
        a = find_centre_mid(contours)
        b = sort_points(a)
        for i in range(n_contours):
            center = (int(b[i,0]), int(b[i,1]))
            cv2.circle(image, center, radius=25, color=(7, 161, 222), thickness=60)
            # center = (int(b[i,0])-20, int(b[i,1])+20)


            text = f"{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            text_color = (255, 255, 255)  # White color
            text_thickness = 7
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    

            # Calculate the position such that the text is centered
            text_x = center[0] - (text_width // 2)
            text_y = center[1] + (text_height // 2)

            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,0,0), text_thickness+8)
            cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

            # cv2.putText(image, f"{i+10}", center, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=12)
            # cv2.putText(image, f"{i+10}", center, cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=7)

        return image

    image_with_contours, mask, contours = segment_trackers(image1, model_path, verbose=verbose)

    # Compute centers
    center_pts = find_centre_mid(contours)
    center_pts_fil = sort_points(center_pts)
    if verbose:
        print(f"Center points \n{center_pts}\n")
        print(f"Filtered centered points \n{center_pts_fil}\n")

    if plot:
        # Plot the centers and tracker numbers. 
        image_with_contours=plot_centers(contours, image_with_contours)
        plt.figure(figsize=(12, 8))
        plt.title('Segmentación de Trackers')
        plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
        plt.show()

    return center_pts_fil, contours

center_pts_fil, contours = tracker_segment_name(image1, model_path, verbose=False, plot=True)
center_pts_fil, contours = tracker_segment_name(image2, model_path, verbose=False, plot=True)

# print(center_pts_fil)



# there might be a problem, if the y value has some error, because if it filters first on th Y value, some values that are to the far right, 
# if they happen to have a lowe y value, they might have a different order. 
# for this, do the clusters algorithm, which, since it has a lot of values, it should be able to decide it well. (FUck this again needs the hyperparameters),
#  which slows down the process. Maybe include the row segmentation as well →→→ I think it is going to be too much computational effort. 

