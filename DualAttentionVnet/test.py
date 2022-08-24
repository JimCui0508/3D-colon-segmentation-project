# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:41:47 2022

@author: Windows
"""



import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob

#import cupy as np
import numpy as np
import random as r
import cv2

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 每个GPU上限控制在90%以内
#session = tf.Session(config=config)



from model_dualattention_vnet3d import createdualattentionnet
path = "/home/public/CTC_ReconResults/zip_files/"


origin_files = glob(path+"/1.3.6.1.4.1.9328.50.4.*/**/ResampledImg.nii.gz", recursive=True)
seg_files = glob(path+"/1.3.6.1.4.1.9328.50.4.*/**/ResampledMask.nii.gz", recursive=True)


print(len(origin_files),len(seg_files))



"""# Convert to Array"""

first_img = io.imread(origin_files[1], plugin="simpleitk")

print(f"shape: {first_img.shape}")
print(f"dtype: {first_img.dtype}")

"""## Visualize Flair Image"""




"""## Visualize Segmented Image"""


seg_img = io.imread(seg_files[1], plugin="simpleitk")



"""# Convert to Array """


def to_array(path, end, label=False):

    # get locations
    files = glob(path+end, recursive=True)
    img_list = []
    
    r.seed(42)
    r.shuffle(files)
    
    for file in files:
        img = io.imread(file, plugin="simpleitk")
        # standardization
    #    img = (img-img.mean())/img.std()
      #  img.astype("float32")
     #   print("img:",img.shape)
        img_list.append(img)
  
    img_list = np.array(img_list)  
    img_list = img_list.reshape((-1,64,128,128))
        
    
  
    return img_list

"""### np.expand_dims()"""





"""# Applying the Function"""

train = to_array(path=path, end="/1.3.6.1.4.1.9328.50.4.*/**/ResampledImg.nii.gz")
print(f"dtype: {train.dtype}")

seg = to_array(path=path, end="/1.3.6.1.4.1.9328.50.4.*/**/ResampledMask.nii.gz", label=True)
print(f"dtype: {seg.dtype}")

train = np.expand_dims(train, axis=4)
print(f"shape: {train.shape}")
seg = np.expand_dims(seg, axis=4).astype(np.float32)
print(f"shape: {seg.shape}")
#train = train.get()
#seg = seg.get()

def dice_loss_3d(Y_gt, Y_pred):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = 1 - tf.reduce_mean(intersection / denominator)
    return loss

def dice_coef(Y_gt, Y_pred):
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    coef = tf.reduce_mean(intersection / denominator)
    return coef

#model = createdualattentionnet(train, 128, 64, 64, 1, True, 0.8, n_class=1)

EPOCHS = 50  # 整个数据集迭代次数
optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.002)

for epoch in range(EPOCHS):
    for i in range(740):  # 一整个数据集分为10个小batch训练
        with tf.GradientTape() as tape:
            prediction = createdualattentionnet(train[i,:,:,:,:], 128, 64, 64, 1, True, 0.8, n_class=1)
            dice_loss = dice_loss_3d(prediction,seg[i,:,:,:,:])
    
   #     trainable_variables = [initial]  # 需优化参数列表
        grads = tape.gradient(dice_loss,tf.trainable_variables())  # 计算梯度
    
        optimizer2.apply_gradients(zip(grads,tf.trainable_variables()))  # 更新梯度
        
    # 每训练完一次，输出一下训练集的准确率    
    accuracy = dice_coef(createdualattentionnet(train, 128, 64, 64, 1, True, 0.9, n_class=1), seg)
    print('Epoch [{}/{}], Train loss: {:.3f}, Test accuracy: {:.3f}'
              .format(epoch+1, EPOCHS, dice_loss, accuracy))


#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=dice_loss_3d, metrics=[dice_coef])
#model.summary()

#model.fit(train, seg, validation_split=0.25, batch_size=4, epochs=100, shuffle=True)

#model.save_weights("subset_model.h5")