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
matplotlib.use('Agg')



MASK_COLORS = [
    "red", "green", "blue",
    "yellow", "magenta", "cyan"
]

# Functions for plotting the predictions
def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    assert(color in MASK_COLORS)
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)

def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask

def get_cmap(arr):
    """[summary]
    
    Args:
        arr (numpy.ndarray): [description]
    
    Returns:
        string: [description]
    """
    if arr.ndim == 3:
        return "gray"
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return "jet"
        elif arr.shape[3] == 1:
            return "gray"

# Metric functions
def iou_score(y_true, y_pred):
    eps = 1.
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = (np.sum(intersection) + eps) / (np.sum(union) + eps)
    return iou

def precision_recall(y_true, y_pred):
    eps = 1.
    intersection = np.logical_and(y_true, y_pred)
    precision = (np.sum(intersection) + eps)/ (np.sum(y_pred) + eps)
    recall = (np.sum(intersection) + eps) / (np.sum(y_true) + eps)
    return precision, recall



if __name__ == '__main__': 

    # Loading model weights and 
    model_filename = '../models/urban_model.h5'

    columns = ['city', 'threshold', 'iou_score', 'precision', 'recall']
    metrics = []
    time_1 = [0]

    urban_list = os.listdir('../input/data/urban/test/images/')
    rural_list = os.listdir('../input/data/rural/test/images/')

    model = satellite_unet(
                input_shape = (256, 256, 3),
                num_classes=1,
                output_activation='sigmoid',
                num_layers=4)
            
    model.load_weights(model_filename)

    # Metrics are calculated and predictions are made for different values of threshold.
    for thresh in [0.05, 0.1, 0.25, 0.5]:
        print('threshold = {}'.format(thresh))
        for j in tqdm(range(len(rural_list))):
            #print('\n', rural_list[j])
            if not os.path.exists('../input/data/rural/test/rural_model_predictions/threshold = {}/{}'.format(thresh, rural_list[j])):
                os.makedirs('../input/data/rural/test/rural_model_predictions/threshold = {}/{}'.format(thresh, rural_list[j]))
            imgs = glob.glob('../input/data/rural/test/images/{}/*.png'.format(rural_list[j]))
            masks = glob.glob('../input/data/rural/test/masks/{}/*.png'.format(rural_list[j]))
            img_list = []
            mask_list = []
            for img in imgs:
                img_list.append(np.array(Image.open(img).resize((256, 256))).astype('float32')/255.0)
            for mask in masks:
                mask_list.append(np.array(Image.open(mask).convert('L').resize((256, 256))).astype('float32')/255.0)
            images_np = np.array(img_list)
            mask_np = np.array(mask_list)
            mask_np = np.expand_dims(mask_np, axis = -1)
            y_pred = model.predict(images_np)
            y_pred = (y_pred > thresh).astype('float')
            city_iou = 0
            city_precision = 0
            city_recall = 0
            
            for i in range(y_pred.shape[0]):
                # Calculating metrics for each image
                iou = iou_score(mask_np[i], y_pred[i])
                precision, recall = precision_recall(mask_np[i], y_pred[i])
                city_iou += iou
                city_precision += precision
                city_recall += recall

                # Overlaying prediction masks on the input images
                plt.axis('off')
                plt.imshow(images_np[i], cmap = get_cmap(images_np[i]))
                plt.imshow(mask_to_rgba(zero_pad_mask(mask_np[i], desired_size=images_np.shape[1]), color = 'green'), cmap = get_cmap(mask_np), alpha = 0.5)
                plt.imshow(mask_to_rgba(zero_pad_mask(y_pred[i], desired_size=images_np.shape[1]), color = 'red'), cmap = get_cmap(y_pred), alpha = 0.5)
                plt.savefig('../input/data/rural/test/rural_model_predictions/threshold = {0}/{1}/{2}.png'.format(thresh, rural_list[j], os.path.splitext(os.path.basename(imgs[i]))[0]), bbox_inches = 'tight', transparent = True, pad_inches = 0)
                plt.close()
                
            city_iou = city_iou / y_pred.shape[0]
            city_precision = city_precision / y_pred.shape[0]
            city_recall = city_recall / y_pred.shape[0]
            metric_dicts = dict.fromkeys(columns)
            metric_dicts['city'] = rural_list[j]
            metric_dicts['threshold'] = thresh
            metric_dicts['iou_score'] = city_iou
            metric_dicts['precision'] = city_precision
            metric_dicts['recall'] = city_recall
            metrics.append(metric_dicts)
    df_scores = pd.DataFrame.from_records(metrics)
    if not os.path.exists('./metrics'):
        os.makedirs('./metrics')
    df_scores.to_csv('./metrics/rural_test_scores.csv')

    