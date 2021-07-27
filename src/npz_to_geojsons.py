# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

# Importing libraries
import math
from unicodedata import name
import rasterio
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from PIL import Image
from shapely.geometry import mapping,Polygon,MultiLineString,LineString
from shapely import wkt
import fiona
from fiona.crs import from_epsg
import time
import shapely
import os
from scipy.sparse import load_npz
from tqdm import tqdm, trange
import multiprocessing as mp
from functools import partial
shapely.speedups.disable()

# Func for converting npz to geojson
def npz_to_geojson(mask, city):
    #print(masks)
    z=17
    n = 2**z
    
    xtile=int(mask.split('\\')[-1].split('.npz')[0].split('.')[-2])
    ytile=int(mask.split('\\')[-1].split('.npz')[0].split('.')[-1])
    
    lon_deg = ((xtile / n)*360.0) - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - (2 * (ytile / n)))))
    lat_deg = lat_rad * (180.0 / math.pi)

    tx=rasterio.transform.from_origin(lon_deg,lat_deg,5.323955725076732e-06 ,5.323955725076732e-06)
    
    image=load_npz(mask).toarray()

    out=find_contours(image,0.5)
    if not out:
        pass
    else:
        
        cs=[]
        fig, ax = plt.subplots()
        for contour in out:  
            cs.append(ax.plot(contour[:, 1], contour[:, 0], linewidth=2))
        
        plt.close()
        poly=[]
        for i in cs:
            
            x=i[0].get_xdata()
            y=i[0].get_ydata()
            aa=rasterio.transform.xy(tx,y,x)
            poly.append(LineString([(i[0], i[1]) for i in zip(aa[0],aa[1])]))
    
            
        list_polygons =  [wkt.loads(p.wkt) for p in poly]
    
        mult=shapely.geometry.MultiLineString(list_polygons)
    
    
        
        crs = from_epsg(4326)
    
        schema = {
            'geometry': 'MultiLineString',
            'properties': {'id': 'int','Name':'str'},
            
        }
        
        
        
        # Write a new Shapefile
        with fiona.open('../inference/geojsons/{}/'.format(city)+mask.split('\\')[-1].split('.npz')[0].split('.')[0]+'.geojson', 'w', 'GeoJSON', schema,crs=crs) as c:
            ## If there are multiple geometries, put the "for" loop here
            #for ls in list_polygons:
                
            c.write({
                'geometry': mapping(mult),
                'properties': {'id': 1,'Name':'Detected Hospital'},
            })

if __name__ == '__main__':
    city_list = os.listdir('../inference/infra_info/predicted/')
    with tqdm(total = len(city_list), desc = 'No. of cities') as pbar1:
        for fname in city_list:
            # Parallelizing by spawning multiple processes
            with mp.Pool(mp.cpu_count() - 11) as pool:
                city_name = os.path.splitext(fname)[0]
                #print(city_name)
                masks = sorted(glob.glob('../inference/preds/{}/*'.format(city_name)), key=lambda x:float(re.findall("(\d+)",x)[0]))
                if not os.path.exists('../inference/geojsons/{}'.format(city_name)):
                    os.makedirs('../inference/geojsons/{}'.format(city_name))
                with tqdm(total = len(masks), desc = '{} progress'.format(city_name)) as pbar2:
                    temp = partial(npz_to_geojson, city = city_name)
                    #print(masks)
                    for i, _ in enumerate(pool.imap_unordered(func = temp, iterable = masks)):
                        pbar2.update()
                os.rename('../inference/infra_info/predicted/{}'.format(fname), '../inference/infra_info/geojsons/{}'.format(fname))
                pbar1.update(1)
            
# Ref: https://github.com/geospoc/rural-school-mapper