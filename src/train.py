# <2021>, by ISB Institute of Data Science
# Contributors: Dr. Shruti Mantri, Gokul S Kumar and Vishal Sriram
# Faculty Mentors: Dr. Manish Gangwar and Dr. Madhu Vishwanathan
# Affiliation: Indian School of Business

# Importing necessary libraries
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
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from tqdm import tqdm
import cv2
import pandas as pd
import time
import tensorflow as tf

# Dir for storng training logs
root_logdir = '.\logs\\rural_model\\'

# Function for getting seperate log directory address for each session
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# Function for plotting the train and validation metrics on the same plot in Tensorboard
class TrainValTensorBoard(TensorBoard):
    
    def __init__(self, log_dir = get_run_logdir(), **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

if __name__ == '__main__':

    if not os.path.exists('./logs/'):
        os.makedirs('./logs/') 
    # Loading images and masks
    masks = sorted(glob.glob("../input/data/rural/train/masks/*.png"))
    imgs = sorted(glob.glob("../input/data/rural/train/images/*.png"))    
    
    images_list = []
    masks_list = []

    # Preprocessing
    print('resizing and normalizing')
    for i in tqdm(range(len(imgs))):    
        images_list.append(np.array(Image.open(imgs[i]).resize((256, 256))).astype('float32')/255.0)
        im = Image.open(masks[i]).convert('L').resize((256, 256))
        masks_list.append(np.array(im).astype('float32')/255.0)

    images_np = np.array(images_list)
    masks_np = np.array(masks_list)

    x = images_np
    y = np.expand_dims(masks_np, axis = -1)
    

    #Splitting into Train and Validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
   
    BatchSize = 16
    # Data augmentation
    train_gen = get_augmented(
        x_train, y_train, batch_size=BatchSize,
        data_gen_args = dict(
            rotation_range=15.,
            horizontal_flip=True,
            vertical_flip=True,
        )) 

    batch = next(train_gen)
    xx, yy = batch
    
    # Importing Satellite Unet model from the keras_unet package
    input_shape = x_train[0].shape
    model = satellite_unet(
        input_shape,
        num_classes=1,
        output_activation='sigmoid',
        num_layers=4
    )

    #Initializing callbacks
    model_filename = '../models/rural_model.h5'
    callback_checkpoint =ModelCheckpoint(
        model_filename, 
        verbose=1, 
        monitor='val_iou',
        mode = 'max',
        save_best_only=True,
    )
    # Learning rate scheduler
    lr_callback = ReduceLROnPlateau(patience = 20)

    #Compiling
    model.compile(
        optimizer=Adam(lr=0.001),
        loss='binary_crossentropy',
        metrics=[iou, iou_thresholded]
    )

    #Training model for 1000 epochs with lr scheduler
    history = model.fit_generator(
        train_gen,
        steps_per_epoch = int(images_np.shape[0] * 3 / BatchSize),
        epochs=1000,
        validation_data=(x_val, y_val),
        callbacks=[lr_callback,callback_checkpoint, TrainValTensorBoard(write_graph = False)]
    )
