#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:02:05 2023

@author: venkat
"""

# Import library
import matplotlib.pyplot as plt
from findpeaks import findpeaks
import findpeaks
import cv2
from skimage.util import random_noise
from photutils.datasets import make_noise_image
import bm3d
# Import image example
img = cv2.imread('/home/venkat/Test_Images/img4.tif', 0)

#img= random_noise(img, mode='speckle', var=0.05, clip=True)

#img= random_noise(img, mode='poisson', seed=100)

#img=make_noise_image(img.shape, distribution='poisson', mean=.05)

# filters parameters
# window size
#winsize = 15
winsize=9
# damping factor for frost
#k_value1 = 2.0
# damping factor for lee enhanced
#k_value2 = 1.0
k_value2=0.1
# coefficient of variation of noise
#cu_value = 0.25
cu_value=0.1
# coefficient of variation for lee enhanced of noise
#cu_lee_enhanced = 0.523
cu_lee_enhanced=0.05
# max coefficient of variation for lee enhanced
#cmax_value = 1.73
cmax_value=.2

# Some pre-processing
# Resize
img = findpeaks.stats.resize(img, size=(300,300))
# Make grey image
img = findpeaks.stats.togray(img)
# Scale between [0-255]
img = findpeaks.stats.scale(img)

plt.figure()

plt.subplot(2,2,1)
# Plot
plt.imshow(img, cmap='gray_r')
plt.title("Image")



image_lee = findpeaks.lee_filter(img, win_size=winsize, cu=cu_value)
plt.subplot(2,2,2)
# Plot
plt.imshow(image_lee, cmap='gray_r')
plt.title("Lee_filter")

plt.subplot(2,2,3)
# Plot
# lee enhanced filter
image_lee_enhanced = findpeaks.lee_enhanced_filter(img, win_size=winsize, k=k_value2, cu=cu_lee_enhanced, cmax=cmax_value)
# Plot
plt.imshow(image_lee_enhanced, cmap='gray_r')

plt.title("Lee_Enhanced Filter")

#bm3d filtering

image_bm3d=bm3d.bm3d(img, sigma_psd=1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# kuan filter
image_kuan = findpeaks.kuan_filter(img, win_size=winsize, cu=cu_value)
plt.subplot(2,2,4)
# Plot
plt.imshow(image_bm3d, cmap='gray_r')
plt.title("bm3d_Filter")