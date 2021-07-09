import pandas as pd
import numpy as np
import os
import shutil

# Loading the rural-urban and train-test indexes
df_train_urban = pd.read_csv('../input/urban_train_index.csv')
df_test_urban = pd.read_csv('../input/urban_test_index.csv')

df_train_rural = pd.read_csv('../input/rural_train_index.csv')
df_test_rural = pd.read_csv('../input/rural_test_index.csv')

# Creating directories for train data and copying the images and masks into it
for i in ['urban', 'rural']:
    for j in ['train', 'test']:
        if not os.path.exists('../input/data/{0}/{1}/masks'.format(i, j)):
            os.makedirs('../input/data/{0}/{1}/masks'.format(i, j))
        if not os.path.exists('../input/data/{0}/{1}/images'.format(i, j)):
            os.makedirs('../input/data/{0}/{1}/images'.format(i, j))        

for i, df in df_train_urban.iterrows():
    shutil.copy(df['filename'], '../input/data/urban/train/images/')
    shutil.copy('../input/masks/' + df['city'] + '/' + df['filename'].split('/')[3].split("\\")[1], '../input/data/urban/train/masks/')

for i, df in df_train_rural.iterrows():
    shutil.copy(df['filename'], '../input/data/rural/train/images/')
    shutil.copy('../input/masks/' + df['city'] + '/' + df['filename'].split('/')[3].split("\\")[1], '../input/data/rural/train/masks/')


urban_list = df_test_urban['city'].unique()
rural_list = df_test_rural['city'].unique()

# Creating directories for test data and copying the images and masks into it
# Test data is stored seperately for each city.
for city in urban_list:
    if not os.path.exists('./data/urban/test/images/{}'.format(city)):
        os.makedirs('./data/urban/test/images/{0}'.format(city))
    if not os.path.exists('./data/urban/test/masks/{0}'.format(city)):
        os.makedirs('./data/urban/test/masks/{0}'.format(city))
    for j, df in df_test_urban[df_test_urban['city'] == city].iterrows():
        shutil.copy(df['filename'], './data/urban/test/images/{}'.format(city))
        shutil.copy('./masks/' + df['city'] + '/' + df['filename'].split('/')[3].split("\\")[1], './data/urban/test/masks/{}'.format(city))

for city in rural_list:
    if not os.path.exists('./data/rural/test/images/{}'.format(city)):
        os.makedirs('./data/rural/test/images/{0}'.format(city))
    if not os.path.exists('./data/rural/test/masks/{0}'.format(city)):
        os.makedirs('./data/rural/test/masks/{0}'.format(city))
    for j, df in df_test_rural[df_test_rural['city'] == city].iterrows():
        shutil.copy(df['filename'], './data/rural/test/images/{}'.format(city))
        shutil.copy('./masks/' + df['city'] + '/' + df['filename'].split('/')[3].split("\\")[1], './data/rural/test/masks/{}'.format(city))


print('urban train:', len(os.listdir('./data/urban/train/images')))
print('urban_test:', len(df_test_urban))

print('rural train:', len(os.listdir('./data/rural/train/images')))
print('rural test: ', len(df_test_rural))


