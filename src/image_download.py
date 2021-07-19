# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

import pandas as pd
import mercantile
import requests
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2

#Create an output folder to store images
# Read POI's from Excel file containing Fields 'Latitude' and 'Longitude'
def mapbox_download(poi,MAT, outpath):
    
    #poi=pd.read_excel("/POI's.xlsx")
    #path to store images
    #print(poi.shape)
    outpath='../input/images/'+outpath+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    stat_list=[]
    
    for index,df in poi.iterrows():

        print(df)
        lat_long=[df['latitude'],df['longitude']]
        top_left = [lat_long[0], lat_long[1]]
        bottom_right = [lat_long[0], lat_long[1]]
        z = 18 # Zoom level

        # We will be appending the centre tile with an extra tile in each direction and also along the corners to avoid the hospital buildings being chaffed off.
        
        #Finding tiles 
        top_left_tiles = mercantile.tile(top_left[1],top_left[0],z)
        bottom_right_tiles = mercantile.tile(bottom_right[1],bottom_right[0],z)
        
        x_tile_range = [top_left_tiles.x - 1,bottom_right_tiles.x + 1]
        y_tile_range = [top_left_tiles.y - 1,bottom_right_tiles.y + 1]

        images_h = []
        for i,x in enumerate(range(x_tile_range[0],x_tile_range[1]+1)):
            images_v = []
            for j,y in enumerate(range(y_tile_range[0],y_tile_range[1]+1)):
                #print(x,y)
                
                r = requests.get('https://api.mapbox.com/v4/mapbox.satellite/'+str(z)+'/'+str(x)+'/'+str(y)+'.pngraw?access_token='+MAT, stream=True)
                if r.status_code == 200:
                    r.raw.decode_content = True
                    image = np.asarray(bytearray(r.raw.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    images_v.append(image)

            # Joining tile horizontally        
            image_col = cv2.vconcat(images_v)
            images_h.append(image_col)
        
        # Joinging tiles horizontally
        image_fin = cv2.hconcat(images_h)
        cv2.imwrite(outpath +str(df['index'])+'.'+str(lat_long[1]) + '.' + str(lat_long[0]) + '.png', image_fin)
        stat_list.append(r.status_code)   
    #print(stat_list)             
    return stat_list    
        

def test_download(address):
    # Reading CSV file containing addresses
    test=pd.read_csv(address, index_col = 0).reset_index()
    outpath = address.split('\\')[1].split('.')[0]
    # Insert Mapbox access token below
    MAT=''
    assert mapbox_download(test,MAT, outpath)==[200]*test.shape[0]
    
if __name__ == '__main__':
    csv_list = glob.glob('../input/addresses/*.csv')
    for i in tqdm(range(len(csv_list))):
        test_download(csv_list[i])





