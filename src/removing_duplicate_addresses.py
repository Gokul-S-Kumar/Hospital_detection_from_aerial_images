# Script for removing the duplicate entries from the downloaded addresses.

import pandas as pd
import glob
import os

if __name__ == '__main__':
    joint = pd.DataFrame()
    length = 0
    
    filenames = glob.glob('./data/address_by_city/*.csv')
    
    for file in filenames:
        df = pd.read_csv(file)
        # Creating a new column to identify the name of the city, this makes it easier to segregate the addresses
        # by cities
        df['city'] = os.path.basename(file).split('.')[0] 
        joint = pd.concat([joint, df])
    
    # Removing the duplicate entries based on place_id.
    joint = joint.drop_duplicates(subset = 'place_id').drop(columns = ['Unnamed: 0'], axis = 1)

    for city in joint['city'].unique():
        df_save = joint[joint['city'] == city]
        df_save.to_csv('./data/addresses/{}.csv'.format(city))