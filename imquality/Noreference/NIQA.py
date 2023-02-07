#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:36:57 2022

@author: venkat
"""
#import PIL.Image
#import cv2
import imquality.brisque as brisque
#import numpy as np
#from skimage.util import random_noise
import skimage.io
import matplotlib.pyplot as plt
import skimage.filters

# Read image


img = skimage.io.imread(fname="/home/venkat/Data Set/1_1.jpg")

sigma = 5.0

# apply Gaussian blur, creating a new image

blurred = skimage.filters.gaussian(img, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.gray()

# Dispaly Original and Blurred images

for a in (ax[0], ax[1]):
       a.axis('off')

ax[0].imshow(img)
ax[0].set_title('Original Data')

ax[1].imshow(blurred)
ax[1].set_title('blurred Image')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()

 # calculation of brisque value
 
print(brisque.score(blurred))

