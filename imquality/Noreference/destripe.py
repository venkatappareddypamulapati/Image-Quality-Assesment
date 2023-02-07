#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:11:45 2023

@author: venkat
"""
from skimage import io
import pystripe
import numpy as np
import cv2
import matplotlib.pyplot as plt
img_path = '/home/venkat/destriping/images.jpeg'
img=cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
print(img.shape)
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
plt.figure()
destriped = pystripe.filter_streaks(img, sigma=[32, 128], level=2, wavelet='db10')
plt.imshow(destriped, cmap='gray')
plt.show()
#im_io=np.hstack((img, destriped)
#cv2.imshow('out_image.png', im_io)

"""from pyvsnr import VSNR
from skimage import io

# read the image to correct
img = io.imread('/home/venkat/destriping/images.jpeg')

# vsnr object creation
vsnr = VSNR(img.shape)

# add filter (at least one !)
vsnr.add_filter(alpha=1e-2, name='gabor', sigma=(1, 30), theta=20)
vsnr.add_filter(alpha=5e-2, name='gabor', sigma=(3, 40), theta=20)

# vsnr initialization
vsnr.initialize()

# image processing
img_corr = vsnr.eval(img, maxit=100, cvg_threshold=1e-4)"""
