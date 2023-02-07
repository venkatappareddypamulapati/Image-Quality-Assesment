#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:51:45 2022

@author: venkat
"""
import numpy as np
 
# sigma(standard deviation) and muu(mean) are the parameters of gaussian
 
def g_filter(kernel_size):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(0.739, 0.4388, kernel_size),
                       np.linspace(0.7329, 0.4338, kernel_size))
    
    dst= (((x) // sigma_x) ** 2)+ (((y) / sigma_y)** 2)
 
    # lower normal part of gaussian
    #normal = 1/(2*np.pi*sigma_x*sigma_y)
    
   # normal=np.sqrtnormal)
 
    # Calculating Gaussian filter
    gauss = np.exp(-dst) 
 
    return gauss/gauss.sum()

kernel_size=7

mx=6
my=-6.138
sigma_x=3.894
sigma_y=6.20

gaussian = g_filter(kernel_size)
print("gaussian filter of {} X {} :".format(kernel_size,kernel_size))
print(gaussian)