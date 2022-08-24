# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:21:09 2022

@author: Windows
"""


import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from keras.utils import multi_gpu_model
#import tensorflow as tf
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob
from Vnet_3d import Vnet_3d
#import cupy as np
import numpy as np
import random as r
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 每个GPU上限控制在90%以内
session = tf.Session(config=config)

def dice_coef(y_true, y_pred):
    smooth = 0.005 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
input_img = Input((128, 128, 128, 1))
model = Vnet_3d(input_img,8,0.2,True)
model.load_weights("vnet_model.h5")
pred_file = "/home/public/CTC_ReconResults/zip_files/ResampledImgv1.nii.gz"

pred_img = io.imread(pred_file, plugin="simpleitk")
pred_img = np.array(pred_img)

expand_img = np.expand_dims(pred_img, axis=0)
expand_img = np.expand_dims(expand_img, axis=-1)

#expand_img = expand_img.get()
pred = model.predict(expand_img)

## save 
out = sitk.GetImageFromArray(pred[0,:,:,:,0])

sitk.WriteImage(out,'/home/public/CTC_ReconResults/zip_files/simpleitk_save.nii.gz')