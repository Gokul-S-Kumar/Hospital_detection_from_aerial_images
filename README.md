![Geospatial Analytics](./Geospatial_Analytics.png "Geospatial Analytics")

## **Contributors:** Dr. Shruti Mantri, Gokul S Kumar, Vishal Siram  
## **Faculty Mentors:** Dr. Manish Gangwar, Dr. Madhu Vishwanathan.

<br/>

## Aim & Objective
To design and develop an intelligent system that identifies hospitals and healthcare facilities from satellite images.

## Steps involved
1. Collecting the names of major cities in Maharashtra.
2. Using Google Maps API to get the addresses of hospitals in these cities, including their coordinates.
3. Using Mapbox API to get the aerial images for the collected coordinates of hospital buildings.
4. Use CVAT Online to annotate these images with the help of Google Earth Pro.
5. Convert the annotations in XML files to binary masks.
6. Split the data into Urban and Rural areas, and then into train and test sets.
7. Split the train data into train and validation data and train the UNet model.
8. Use the model weights to make predictions.


## Repo contents
- **input**: The address (and coordinates) CSV's from Google Maps API, the image files from Mapbox and the respective annotation XML files from CVAT.
- **models**: The final model weights.
- **notebooks**: Jupyter notebooks used for data visualization and other purposes.
- **src**: All the python scripts associated with the project.
- **inference**: Contains the geotiff infra maps, tile numbers containing infrastructure, images used for inference & predictions in .npz and geojson format.

## FAQs

1. How to clone the repo to start developing?  
A: ```git clone https://github.com/Gokul-S-Kumar/Hospital_detection_from_aerial_images```

2. How to set-up the dev environment?  
A: Use the "requirements.txt" file to clone the environment in which the model was developed.

3. How to build on the existing work?  
A: Use the model weights available [here](https://github.com/Gokul-S-Kumar/Hospital_detection_from_aerial_images/tree/main/models) as pre-trained weights and develop new strategies to further improve the predictions.


