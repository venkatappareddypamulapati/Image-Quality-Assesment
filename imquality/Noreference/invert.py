#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:09:58 2022

@author: venkat
"""

import cv2

#%%
img = cv2.imread("/home/venkat/image-quality/imquality/Noreference/Lenna.png")
img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
cv2.imshow("Pic",img1)

#%%

Gaussian= cv2.GaussianBlur(img1,(9,9),0)

Mask1 =img1 - Gaussian
UnsharpImage = img1 + (Mask1)

#%%

cv2.imshow("UnsharpImage",UnsharpImage)

#%%
img_not = cv2.bitwise_not(UnsharpImage)

cv2.imshow("Invert1",img_not)
