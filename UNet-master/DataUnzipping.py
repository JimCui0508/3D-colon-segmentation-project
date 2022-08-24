# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:34:22 2022

@author: Windows
"""
import os

import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob

import numpy as np
import zipfile

path = "/home/public/CTC_ReconResults/zip_files/"


zip_files = glob(path+"/*.zip", recursive=False)


for zip_file in zip_files:    
    with zipfile.ZipFile(zip_file, "r") as z_fp:
        z_fp.extractall(path)

