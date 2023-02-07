#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:53:02 2022

@author: venkat
"""

import cv2
	 
	# Reading the image
image = cv2.imread('/home/venkat/Data Set/img1.png')

	 

	# dividing height and width by 2 to get the center of the image
height, width, shape = image.shape[:]

	# get the center coordinates of the image to create the 2D rotation matrix
center = (width/2, height/2)

	 
	# using cv2.getRotationMatrix2D() to get the rotation matrix

rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)


	# rotate the image using cv2.warpAffine
    
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

	 
cv2.imshow('Original image', image)

cv2.imshow('Rotated image', rotated_image)

	# wait indefinitely, press any key on keyboard to exit
cv2.waitKey(0)
	# save the rotated image to disk
cv2.imwrite('rotated_image.png', rotated_image)