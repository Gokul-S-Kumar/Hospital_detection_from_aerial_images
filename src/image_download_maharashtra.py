# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

# Importing libraries
from functools import partial
import pandas as pd
import requests
import os
from tqdm import tqdm
import numpy as np
import cv2
import multiprocessing as mp

# Mapbox access token
MAT=''

# Function for downloading mapbox raster tile
def mapbox_download(tile, city):
    z = 17 # Zoom level
    
    outpath='../inference/images/'+city+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    try:
        r = requests.get('https://api.mapbox.com/v4/mapbox.satellite/'+str(z)+'/'+str(tile[0])+'/'+str(tile[1])+'.pngraw?access_token='+MAT, stream=True)
        if r.status_code == 200:
            r.raw.decode_content = True
            image = np.asarray(bytearray(r.raw.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite(outpath +str(tile[2])+'.'+str(tile[0]) + '.' + str(tile[1]) + '.png', image)
    except:
        print('Crashed while {0} in {1},{2}'.format(city, tile[0], tile[1]))   
    
if __name__ == '__main__':
    csv_list = [f for f in os.listdir('../inference/infra_info/') if f.endswith('.csv')]
    with tqdm(total = len(csv_list), desc = 'No. of cities') as pbar1:
        for city in csv_list:
            df = pd.read_csv('./infra_info/{}'.format(city))
            df_infra = df[df['infra'] == True].reset_index().drop('index', axis = 1)
            xtile = df_infra['xtile'].to_list()
            ytile = df_infra['ytile'].to_list()
            count = df_infra.index.to_list()
            # Parallelizing the image download with multiple processes
            with mp.Pool(mp.cpu_count() - 11) as pool:
                with tqdm(total = df_infra.shape[0], desc = '{} Progress'.format(os.path.splitext(city)[0])) as pbar2:
                    temp = partial(mapbox_download, city = os.path.splitext(city)[0])
                    for i, _ in enumerate(pool.imap_unordered(temp, iterable = zip(xtile, ytile, count))):
                        pbar2.update()
            os.rename('../inference/infra_info/{}'.format(city), '../inference/infra_info/downloaded/{}'.format(city))
            pbar1.update(1)



