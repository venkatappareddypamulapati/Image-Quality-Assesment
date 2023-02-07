#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:43:50 2022

@author: venkat
"""
import numpy as np
import math

def rotate(x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr, yr

[xr, yr]=rotate(2,3, 90, 0)

print('New Coordinates (xr, yr): %d, %d' %((xr, yr)))


#%%
R = 4
C = 4
  
# Function to rotate the matrix by 180 degree
def reverseColumns(arr):
    for i in range(C):
        j = 0
        k = C-1
        while j < k:
            t = arr[j][i]
            arr[j][i] = arr[k][i]
            arr[k][i] = t
            j += 1
            k -= 1
 #%%            
# Function for transpose of matrix
def transpose(arr):
    for i in range(R):
        for j in range(i, C):
            t = arr[i][j]
            arr[i][j] = arr[j][i]
            arr[j][i] = t
  
# Function for display the matrix
def printMatrix(arr):
    for i in range(R):
        for j in range(C):
            print(arr[i][j], end = " ");
        print();
#%%  
# Function to anticlockwise rotate matrix
# by 180 degree
def rotate180(arr):
    transpose(arr);
    reverseColumns(arr);
    transpose(arr);
    reverseColumns(arr);
  
# Driven code
arr = [ [ 1, 2, 3, 4 ],
        [ 5, 6, 7, 8 ],
        [9, 10, 11, 12 ],
        [13, 14, 15, 16 ] ];
rotate180(arr);
printMatrix(arr);
#%% Image/Kernel Rotation

import cv2
# Reading the image
kernel = [ [6, 7, 8 ],
        [10, 11, 12 ],
        [14, 15, 16 ],
        [1,2,3]];
  
# Extracting height and width from 
kernel=np.array(kernel)

kernel=kernel.astype(np.float32)

height, width = kernel.shape[:2]
  
# get the center coordinates of the
# image to create the 2D rotation
# matrix
center = (height/2, width/2)
  
# using cv2.getRotationMatrix2D() 
# to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-180, scale=1)
  
# rotate the image using cv2.warpAffine 
# 90 degree anticlockwise
rotated_kernel = cv2.warpAffine(
    src=kernel, M=rotate_matrix, dsize=(width, height))
  
print(rotated_kernel)


#%%
import rotate_matrix
  
mat = [[5, 2, 6], [8, 2, 9], [3, 6, 7], [1, 2, 1]]
  
print(rotate_matrix.clockwise(mat))

print(rotate_matrix.anti_clockwise(mat))
