#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:32:42 2022

@author: venkat
"""
import numpy as np
import cv2
from skimage import color, data, restoration
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from skimage.util import img_as_ubyte
from mpl_toolkits import mplot3d


def g_filter(kernel_size, sigma_x, mx, sigma_y, my):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
   
   
    
    dst= (((x-mx) // sigma_x) ** 2)+ (((y-my) // sigma_y)** 2)
 
    # lower normal part of gaussian
    normal = 1/(2*np.pi*sigma_x*sigma_y)
    
    #normal=np.sqrt(normal)
 
    # Calculating Gaussian filter
    gauss = normal*np.exp(-dst) 
 
    return gauss/gauss.sum()

kernel_size=3

#x, y = np.meshgrid(np.linspace(0,18, kernel_size),
           #    np.linspace(0.1988, 0.0502, kernel_size))
 

x=np.linspace(-9,9, kernel_size)
    
y=np.linspace(-9,9, kernel_size)
    
y=np.array(y)

sigma_y=np.std(y)
     
x,y=np.meshgrid(x,y)

gaussian = g_filter(kernel_size, sigma_x=np.std(x), mx=np.average(x), sigma_y=np.std(y), my=np.average(y))

#gaussian=np.array(gaussian, dtype='float64')

print("gaussian filter of{} X {} :".format(kernel_size,kernel_size))

print(gaussian)
