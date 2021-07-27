# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

#Importing libraries
import mercantile
import pandas as pd
import numpy as np
from tqdm import tqdm
import rasterio
import os
from osgeo import gdal

# Func for getting the infrastructure info from each tif image.
def get_infra_info(file):
	ds = gdal.Open('../inference/tif_files/{}'.format(file))
	width = ds.RasterXSize
	height = ds.RasterYSize
	# Getting the coordinate info from the geotiff file.
	gt = ds.GetGeoTransform()
	minx = gt[0]
	miny = gt[3] + width*gt[4] + height*gt[5]
	maxx = gt[0] + width*gt[1] + height*gt[2]
	maxy = gt[3]

	# Reading the geotiff file to identifying the band 1 (infrastructure) areas
	band_id = 1
	raster = rasterio.open('../inference/tif_files/{}'.format(file))
	band_arr = raster.read(band_id)
	# 0.0025 is the latitude and longitude span for each zoom 17 tile.
	tile_span = 0.0025/((maxx-minx)/band_arr.shape[1])

	# Getting the array dimensions for Zoom 17 tiles
	x_tile_span = band_arr.shape[1] / int(tile_span)
	y_tile_span = band_arr.shape[0] / int(tile_span)

	y_split = np.array_split(band_arr, int(y_tile_span))
	
	# Extracting zoom 17 tiles having infrastructure mapping.
	df_dict = {}
	lat = maxy
	span = (maxx - minx)/width
	count = 0
	for i in y_split:
		x_split = np.array_split(i, int(x_tile_span), axis = 1)
		long = minx
		for j in x_split:
			tile = mercantile.tile(long, lat, 17)
			tilex = tile.x
			tiley = tile.y
			if np.any(j == 1):
				df_dict[count] = {'xtile' : tilex, 'ytile' : tiley, 'infra' : True}
				count += 1
			else:
				df_dict[count] = {'xtile' : tilex, 'ytile' : tiley, 'infra' : False}
				count += 1
			long += span * j.shape[1]
		lat -= span * i.shape[0]
	return df_dict 


if __name__ == '__main__':
	tif_list = os.listdir('../inference/tif_files/')
	# Getting the tile numbers of tiles having infrastructure mapping
	for file in tqdm(range(len(tif_list))):
		city = os.path.splitext(tif_list[file])[0]
		df_dict = get_infra_info(tif_list[file])
		df = pd.DataFrame.from_dict(df_dict, 'index')
		df.to_csv('../inference/infra_info/{}.csv'.format(city), index = False)
		
		
	
		

	

