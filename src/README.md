# Data Pipeline, Training, and Inference.

<2021>, by ISB Institute of Data Science  
Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram  
Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan  
Affiliation: Indian School of Business  

## Description
Run the scripts in the following order.
1. **getting_addresses_from_google.py** - Script for getting the addresses and coordinates of hospital in each city listed in "../input/maharashtra_cities.csv" file.
2. **removing_duplicate_addresses.py** - Script for removing any duplicate entries in the addresses downloaded earlier.
3. **image_download.py** - Script for downloading the hospital images in each city as per the coordinates obtained earlier.
4. Use CVAT to label the images with the help of GEP and store the "annotations.xml" file in "../input/annotations", in seperate folders for each city. We have only considered major hospitals, hospital buildings that are located on the side of a major road and also buildings that are distinguishable from the surroundings. The building images which do not match the above criteria are not labelled and will not be considered for training.
5. **generate_binary_masks.py** - Script for generating binary masks from the annotation XML files. The images are seperated into trainable and non-trainable
6. **rural_urban_train_test_index.py** - Script for splitting the entire trainable data into rural and urban zones and further into train and test sets in each zone.
7. **data_split.py** - Script for creating rural, urban and further train and test directories and storing the corresponding data-points accordingly.
8. **train.py** - Training script. Change the input and output directories to "rural" and "urban" as per the need.
9. **inference.py** - Script for creating predictions and calculating test set metrics. Use the respective model weights for rural and urban areas. The green mask in the prediction refers to ground truth and the red mask is the prediction by the model. The metrics are calculated for each city.
10. **info_from_tifs.py** - Script for extracting the zoom 17 tiles in Maharashtra that contains infrastructure.
11. **image_download_maharashtra.py** - Script for downloading the tiles extracted in info_from_tiles.py script.
12. **inference_maharashtra.py** - Inference script for getting predictions on the downloaded images in .npz format.
13. **npz_to_geojson.py** - Script for converting the .npz prediction files to geojson format.