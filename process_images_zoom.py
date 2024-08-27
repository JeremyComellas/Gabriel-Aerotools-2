
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display an image using Matplotlib
def mostrar(image, title='Image', folder="", name="image", save=False):
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    # plt.minorticks_on()
    # plt.grid(which='both', color='red', linestyle='-', linewidth=0.4, alpha=0.7)

    # plt.axis('off')
    if save:
        plt.savefig(folder+name)
    else:   
        plt.show()
    
def mostrar2(image1, image2, title='Image'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    # ax[0].grid('red')
    ax[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # ax[1].grid('red')
    fig.suptitle(title)
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


    folder_path_1 = 'Data/Fotos_RGB/'
    folder_path_2 = 'Data/Fotos_Termo/'
    image_names_1 = get_picture_filenames(folder_path_1)
    image_names_2 = get_picture_filenames(folder_path_2)

    cual = image_num    # 479  # 363-396       it moves to the left, so picture i+1 is equivalent to shifting picture i to the right. # Im not that sure about this anymore
    # Comparison between images
    image1_s = 'Data/Fotos_RGB/' + image_names_1[cual]
    # image2 = 'Data/Fotos_RGB/' + image_names_1[cual-1]
    # print(f"Image read is: \n{image1_s}")#\n{image2}")
    image1 = cv2.imread(image1_s)
    # image2 = cv2.imread(image2)
    image1 = zoom_in(image1, zoom_fact)
    # image2 = zoom_in(image2, zoom_fact)


    return image1, image1_s.split("/")[-1].strip(".JPG") #, image2

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
    # print(image.shape)
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




# image_num = 56
# image1, image1_s = load_images_inv(image_num, zoom_fact=1.6)

# print(image1_s)

# # Save the image in good quality, without plot
# cv2.imwrite(f'Data_processed/{image1_s}__Z.jpg', image1)




names = get_picture_filenames('Data/Fotos_RGB/')
print(len(names))

ini = 700
fin = 1000
for i in range(ini, fin+1):
    image_num = i
    print((i-ini)/(fin-ini)*100)
    image1, image1_s = load_images_inv(image_num, zoom_fact=1.6)
    image1 = un_distort(image1, 300)
    # print(image1_s)

    # Save the image in good quality, without plot
    cv2.imwrite(f'Data_processed/Undistorted_4/{image1_s}__Z.jpg', image1)