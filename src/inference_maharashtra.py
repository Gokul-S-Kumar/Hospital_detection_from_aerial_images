# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
from keras_unet.utils import plot_imgs
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.utils import plot_imgs
from keras_unet.models import satellite_unet
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.metrics import MeanIoU
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from tqdm import tqdm
import cv2
import pandas as pd
from keras import backend as K
import time
import matplotlib
import h5py
from scipy.sparse import csr_matrix, save_npz
import multiprocessing as mp
from functools import partial
import tensorflow as tf
matplotlib.use('Agg')



# Func for inference
def predict(image, city, model):
    img = np.array(Image.open('../inference/images/{0}/{1}'.format(city, image))).astype('float32')/255.0
    img = np.expand_dims(img, axis = 0)
    #print(img.shape)
    y_pred = model.predict(img)
    y_pred = (y_pred > thresh).astype('float')
    y_pred = np.squeeze(y_pred[0], axis = 2)
    y_pred_sparse = csr_matrix(y_pred)
    save_npz('../inference/preds/{0}/{1}.npz'.format(city, os.path.splitext(image)[0]), y_pred_sparse)

if __name__ == '__main__': 
 
    rural_model_filename = '../models/rural_model.h5'
    urban_model_filename = '../models/urban_model.h5'

    urban_dists = ['Thane', 'Pune', 'Mumbai Suburban', 'Nashik', 'Nagpur',
       'Ahmednagar', 'Solapur', 'Jalgaon', 'Kolhapur', 'Aurangabad',
       'Nanded', 'Mumbai City', 'Satara', 'Palghar', 'Amravati', 'Sangli',
       'Yavatmal', 'Raigad', 'Beed']
    
    rural_dists = ['Buldana', 'Latur', 'Chandrapur', 'Dhule', 'Jalna', 'Parbhani',
       'Akola', 'Osmanabad', 'Nandurbar', 'Ratnagiri', 'Gondia', 'Wardha',
       'Bhandara', 'Washim', 'Hingoli', 'Gadchiroli', 'Sindhudurg']

    city_list = os.listdir('../inference/infra_info/downloaded/')
    
    with tqdm(total = len(city_list), desc = 'No. of cities: ') as pbar1:
        for city in city_list:
            city_name = os.path.splitext(city)[0]
            # Checking if the area belong to rural or urban zone.
            if city_name in urban_dists:
                model = satellite_unet(
                    input_shape = (256, 256, 3),
                    num_classes=1,
                    output_activation='sigmoid',
                    num_layers=4)
                model.load_weights(urban_model_filename)
        
            else:
                model = satellite_unet(
                    input_shape = (256, 256, 3),
                    num_classes=1,
                    output_activation='sigmoid',
                    num_layers=4)
                model.load_weights(rural_model_filename)

            image_list = os.listdir('../inference/images/{}/'.format(city_name))
            if not os.path.exists('../inference/preds/{}'.format(city_name)):
                os.makedirs('../inference/preds/{}'.format(city_name))

            thresh = 0.25
            
            with tqdm(total = len(image_list), desc  = '{} progress: '.format(city_name)) as pbar2:
                for image in image_list:
                    predict(image, city_name, model)
                    pbar2.update(1)
            
            os.rename('../inference/infra_info/downloaded/{}'.format(city), '../inference/infra_info/predicted/{}'.format(city))

            pbar1.update(1)