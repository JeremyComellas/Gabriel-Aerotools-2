# Save the images onto a sepparate folder
# has to be run 3 times, one for __0, one for __1, and one for __2. 

# code to access a single row of trackers. selecting 0, 1 or 2:

import os
import cv2

def get_files_with_suffix(directory, suffix):
    # List to hold the matching file names
    matching_files = []
    if suffix == 0:
        suffix = '__0.jpg'
    elif suffix == 1:
        suffix = '__1.jpg'
    elif suffix == 2:
        suffix = '__2.jpg'

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename ends with the specified suffix
        if filename.endswith(suffix):
            matching_files.append(filename)

    return matching_files

# Example usage
directory_path = 'Data_processed/Split_second/'  # Replace with the path to your directory
suffix = 0

# Get all files ending with __0.jpg
files_ending_with_suffix = get_files_with_suffix(directory_path, suffix)


saving_path = f'Data_processed/Split_ini/__{str(suffix)}/'
# Print the matching files
i = 0
for file in files_ending_with_suffix:
    # print(type(file))
    image = cv2.imread(directory_path+file)
    saver = saving_path + file
    cv2.imwrite(saver, image)
    print(f"Succesfully re-arranged {i/len(files_ending_with_suffix)*100:.2f}%")
    i += 1
