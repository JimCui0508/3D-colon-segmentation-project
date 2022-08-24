# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:16:55 2022

@author: Windows
"""
import tensorflow as tf
from keras.models import Model
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob
import keras.backend.tensorflow_backend as KTF
import cupy as np
from keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 每个GPU上限控制在90%以内
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)
def dice_coef(y_true, y_pred):
    smooth = 0.005 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
model = tf.keras.models.load_model('3DUnet.h5',custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef':dice_coef})


pred_file = "/home/public/CTC_ReconResults/zip_files/ResampledImg.nii.gz"

pred_img = io.imread(pred_file, plugin="simpleitk")
pred_img = np.array(pred_img)

expand_img = np.expand_dims(pred_img, axis=0)
expand_img = np.expand_dims(expand_img, axis=-1)

expand_img = expand_img.get()
pred = model.predict(expand_img)

## save 
out = sitk.GetImageFromArray(pred[0,:,:,:,0])

sitk.WriteImage(out,'/home/public/CTC_ReconResults/zip_files/simpleitk_save.nii.gz')