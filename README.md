# 3DUNet and VNet - Keras

*GMine Cooperation** - *"3D Convolutional Neural Network for colon Segmentation"*

Gmine's gut segmentation project, which may be used for tumor prediction.
The data set used is obtained from Academician Wang of Friendship Hospital.
At present, the project has completed the development and debugging of 3DUNet and 3DVNet magic, and the model performance is shown in the diagram in the project.
At present, about 1000 intestinal CT data are used in the project. Computational conformal geometry algorithm is used to annotate the data, and then supervised model training is used to solve the problem of image annotation difficulty in medical image processing, and its effect is better than the traditional geometric segmentation method.

# What I Learned

* Automatic pre-processing algorithm Medical Images in the NIfTI format with simpleitk, scikit-image.
* Successfully adapted [VNet](https://arxiv.org/abs/1606.04797 "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation") (3D CNN Architecture) to Tensorflow/Keras.
