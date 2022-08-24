# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:03:21 2022

@author: Windows
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Input, UpSampling2D,BatchNormalization
from keras.optimizers import adam_v2
import os
import skimage.io as io
from glob import glob
import numpy as np
import random as r
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
#config = tf.ConfigProto()
#session = tf.Session(config=config)
path = "/home/public/CTC_ReconResults/zip_files/"


num_classes = 2
input_shape = (128, 128, 128, 1)
patch_size = (2, 2, 2)  # 2-by-2 sized patches
dropout_rate = 0.05  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 32  # Initial image size

num_patch_x = input_shape[0] // patch_size[0] 
num_patch_y = input_shape[1] // patch_size[1]
num_patch_z = input_shape[2] // patch_size[2]

learning_rate = 1e-3
batch_size = 128
num_epochs = 5
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

"""## Helper functions

We create two helper functions to help us get a sequence of
patches from the image, merge patches, and apply dropout.
"""
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
    img_list = img_list.reshape((-1,128,128,128))
    return img_list


def window_partition(x, window_size):
    _,depth, height, width, channels = x.shape
    patch_num_z = depth // window_size
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_z, window_size, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, depth, height, width, channels):
    patch_num_z = depth // window_size
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_z, window_size, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 5, 2, 4, 6, 7))
    x = tf.reshape(x, shape=(-1, depth, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

"""## Window based multi-head self-attention

Usually Transformers perform global self-attention, where the relationships between
a token and all other tokens are computed. The global computation leads to quadratic
complexity with respect to the number of tokens. Here, as the [original paper](https://arxiv.org/abs/2103.14030)
suggests, we compute self-attention within local windows, in a non-overlapping manner.
Global self-attention leads to quadratic computational complexity in the number of patches,
whereas window-based self-attention leads to linear complexity and is easily scalable.
"""

class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        ) * (2 * self.window_size[2] - 1) 
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        coords_matrix = np.meshgrid(coords_d, coords_h, coords_w)
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(3, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k


        num_window_elements = self.window_size[0] * self.window_size[1] * self.window_size[2]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(3, 0, 1, 2))
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

"""## The complete Swin Transformer model

Finally, we put together the complete Swin Transformer by replacing the standard multi-head
attention (MHA) with shifted windows attention. As suggested in the
original paper, we create a model comprising of a shifted window-based MHA
layer, followed by a 2-layer MLP with GELU nonlinearity in between, applying
`LayerNormalization` before each MSA layer and each MLP, and a residual
connection after each of these layers.

Notice that we only create a simple MLP with 2 Dense and
2 Dropout layers. Often you will see models using ResNet-50 as the MLP which is
quite standard in the literature. However in this paper the authors use a
2-layer MLP with GELU nonlinearity in between.
"""

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            depth, height, width = self.num_patch
            d_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, depth, height, width, 1))
            count = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        mask_array[:, d, h, w, :] = count
                        count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size* self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        depth, height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, depth, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size,-self.shift_size, -self.shift_size], axis=[1, 2, 3]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, depth, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size, self.shift_size], axis=[1, 2, 3]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, depth * height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

"""## Model training and evaluation

### Extract and embed patches

We first create 3 layers to help us extract, embed and merge patches from the
images on top of which we will later use the Swin Transformer class we built.
"""

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_z = patch_size[0]
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            input=images,
            ksizes=(1, self.patch_size_z, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_z, self.patch_size_x, self.patch_size_y, 1),
           # rates=(1, 1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        depth, height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, depth, height, width, C))
        x0 = x[:,0::2 ,0::2, 0::2, :]
        x1 = x[:,1::2 ,0::2, 0::2, :]
        x2 = x[:,0::2 ,1::2, 0::2, :]
        x3 = x[:,1::2 ,1::2, 0::2, :]
        x4 = x[:,0::2 ,0::2, 1::2, :]
        x5 = x[:,1::2 ,0::2, 1::2, :]
        x6 = x[:,0::2 ,1::2, 1::2, :]
        x7 = x[:,1::2 ,1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3, x4, x5, x6, x7), axis=-1)
        x = tf.reshape(x, shape=(-1, (depth // 2), (height // 2), (width // 2), 8 * C))
        return self.linear_trans(x)


"""### Build the model

We put together the Swin Transformer model.
"""

input = layers.Input(input_shape)
#x = layers.RandomCrop(image_dimension, image_dimension)(input)
#x = layers.RandomFlip("horizontal")(x)
x = PatchExtract(patch_size)(input)
x = PatchEmbedding(num_patch_z * num_patch_x * num_patch_y, embed_dim)(x)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_z, num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_z, num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)
x = PatchMerging((num_patch_z, num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
conv1 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (x)
batch1 = BatchNormalization()(conv1)
conv1 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (batch1)
batch1 = BatchNormalization()(conv1)
pool1 = MaxPooling3D(pool_size=2)(batch1)

conv2 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (pool1)
batch2 = BatchNormalization()(conv2)
conv2 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (batch2)
batch2 = BatchNormalization()(conv2)
pool2 = MaxPooling3D(pool_size=2)(batch2)

conv3 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (pool2)
batch3 = BatchNormalization()(conv3)
conv3 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (batch3)
batch3 = BatchNormalization()(conv3)
pool3 = MaxPooling3D(pool_size=2)(batch3)

conv4 = Conv3D(512, kernel_size=3, activation='relu', padding='same') (pool3)
batch4 = BatchNormalization()(conv4)
conv4 = Conv3D(512, kernel_size=3, activation='relu', padding='same') (batch4)
batch4 = BatchNormalization()(conv4)
pool4 = MaxPooling3D(pool_size=2)(batch4)

conv5 = Conv3D(1024, kernel_size=3, activation='relu', padding='same') (pool4)
batch5 = BatchNormalization()(conv5)
conv5 = Conv3D(1024, kernel_size=3, activation='relu', padding='same') (batch5)
batch5 = BatchNormalization()(conv5)

up6 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch5)
up6 = concatenate([up6, conv4], axis=4)
conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same') (up6)
batch6 = BatchNormalization()(conv6)
conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same') (batch6)
batch6 = BatchNormalization()(conv6)

up7 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch6)
up7 = concatenate([up7, conv3], axis=4)
conv7 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (up7)
batch7 = BatchNormalization()(conv7)
conv7 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (batch7)
batch7 = BatchNormalization()(conv7)

up8 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch7)
up8 = concatenate([up8, conv2], axis=4)
conv8 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (up8)
batch8 = BatchNormalization()(conv8)
conv8 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (batch8)
batch8 = BatchNormalization()(conv8)

up9 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch8)
up9 = concatenate([up9, conv1], axis=4)
conv9 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (up9)
batch9 = BatchNormalization()(conv9)
conv9 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (batch9)
batch9 = BatchNormalization()(conv9)

output = Conv3D(1, kernel_size=1, activation='sigmoid')(batch9)

train = to_array(path=path, end="/1.3.6.1.4.1.9328.50.4.*/**/ResampledImgv2.nii.gz")
print(f"dtype: {train.dtype}")
train = np.expand_dims(train, axis=4)
seg = to_array(path=path, end="/1.3.6.1.4.1.9328.50.4.*/**/ResampledMaskv2.nii.gz", label=True)
print(f"dtype: {seg.dtype}")
seg = seg.astype("float32")


'''
input = layers.Input((128,128,128,1))
conv1 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (input)
batch1 = BatchNormalization()(conv1)
conv1 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (batch1)
batch1 = BatchNormalization()(conv1)
pool1 = MaxPooling3D(pool_size=2)(batch1)

conv2 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (pool1)
batch2 = BatchNormalization()(conv2)
conv2 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (batch2)
batch2 = BatchNormalization()(conv2)
pool2 = MaxPooling3D(pool_size=2)(batch2)

conv3 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (pool2)
batch3 = BatchNormalization()(conv3)
conv3 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (batch3)
batch3 = BatchNormalization()(conv3)
pool3 = MaxPooling3D(pool_size=2)(batch3)

conv4 = Conv3D(512, kernel_size=3, activation='relu', padding='same') (pool3)
batch4 = BatchNormalization()(conv4)
conv4 = Conv3D(512, kernel_size=3, activation='relu', padding='same') (batch4)
batch4 = BatchNormalization()(conv4)
pool4 = MaxPooling3D(pool_size=2)(batch4)

conv5 = Conv3D(1024, kernel_size=3, activation='relu', padding='same') (pool4)
batch5 = BatchNormalization()(conv5)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_z, num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=0,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(batch5)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_z, num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)
conv5 = Conv3D(1024, kernel_size=3, activation='relu', padding='same') (x)
batch5 = BatchNormalization()(conv5)

up6 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch5)
up6 = concatenate([up6, conv4], axis=4)
conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same') (up6)
batch6 = BatchNormalization()(conv6)
conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same') (batch6)
batch6 = BatchNormalization()(conv6)

up7 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch6)
up7 = concatenate([up7, conv3], axis=4)
conv7 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (up7)
batch7 = BatchNormalization()(conv7)
conv7 = Conv3D(256, kernel_size=3, activation='relu', padding='same') (batch7)
batch7 = BatchNormalization()(conv7)

up8 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch7)
up8 = concatenate([up8, conv2], axis=4)
conv8 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (up8)
batch8 = BatchNormalization()(conv8)
conv8 = Conv3D(128, kernel_size=3, activation='relu', padding='same') (batch8)
batch8 = BatchNormalization()(conv8)

up9 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same') (batch8)
up9 = concatenate([up9, conv1], axis=4)
conv9 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (up9)
batch9 = BatchNormalization()(conv9)
conv9 = Conv3D(64, kernel_size=3, activation='relu', padding='same') (batch9)
batch9 = BatchNormalization()(conv9)

output = Conv3D(1, kernel_size=1, activation='sigmoid')(batch9)
'''
from keras import backend as K
def dice_coef(y_true, y_pred):
    smooth = 0.005 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model = keras.Model(input, output)
model.summary()
model.compile(
    loss=dice_coef_loss,
    optimizer=adam_v2.Adam(learning_rate=1e-3),
    metrics=[
        dice_coef
    ]
)
model.fit(train, seg, validation_split=0.25, batch_size=8, epochs=100, shuffle=True)