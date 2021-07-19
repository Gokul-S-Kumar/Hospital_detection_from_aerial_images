# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os
import shutil

# Loading the list of cities
df_city = pd.read_csv('../input/maharashtra_cities.csv')

# Splitting into rural and urban zones
urban_list = df_city.loc[:19, 'City'].to_list()

rural_list = df_city.loc[20:, 'City'].to_list()

file_list = os.listdir('../input/masks/')


df_files = pd.DataFrame(columns = ['city', 'filename', 'zone'])

# Splitting the labelled images into urban and rural sets
for city in file_list:
    for filename in glob.glob('../input/images/{}/trainable/*.png'.format(city)):
        if city in urban_list:
            zone = 'urban'
        else:
            zone = 'rural'
        df_files.loc[len(df_files)] = [city, filename, zone]


# Splitting the urban and rural sets further to train and test sets in each zone.
urban_train_files, urban_test_files = train_test_split(df_files[df_files['zone'] == 'urban'], stratify = df_files[df_files['zone'] == 'urban']['city'], test_size = 0.2, random_state = 42)
rural_train_files, rural_test_files = train_test_split(df_files[df_files['zone'] == 'rural'], stratify = df_files[df_files['zone'] == 'rural']['city'], test_size = 0.2, random_state = 42)

urban_train_files.to_csv('../input/urban_train_index.csv')
urban_test_files.to_csv('../input/urban_test_index.csv')


rural_train_files.to_csv('../input/rural_train_index.csv')
rural_test_files.to_csv('../input/rural_test_index.csv')

