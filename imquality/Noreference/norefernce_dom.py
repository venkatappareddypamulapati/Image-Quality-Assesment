#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:22:52 2022

@author: venkat
"""

import numpy as np

# Computer Vision Library
import cv2


import matplotlib.pyplot as plt

#Base Image Load


def load(imagePath, blur=False, blur_size=(5,5)):    
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    
    # Add Gaussian Blur
    if blur:
        image = cv2.GaussianBlur(image, blur_size, cv2.CV_32F)
    
    Im = cv2.medianBlur(image, 3, cv2.CV_64F).astype("double")/255.0
    return image, Im

# Method to display image matrix
def showImage(image, ax, fig, title, cmap=None):
    im = ax.imshow(image, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    
imagePath = "/home/venkat/test_project/img.jpeg"
image, Im = load(imagePath)   

#Sharpness Estimation

def dom(median_blur_image):

    median_shift_up = np.pad(median_blur_image, ((0,2), (0,0)), 'constant')[2:,:]
    median_shift_down = np.pad(median_blur_image, ((2,0), (0,0)), 'constant')[:-2,:]
    domx = np.abs(median_shift_up - 2*median_blur_image + median_shift_down)
    
    median_shift_left = np.pad(median_blur_image, ((0,0), (0,2)), 'constant')[:,2:]
    median_shift_right = np.pad(median_blur_image, ((0,0), (2,0)), 'constant')[:,:-2]
    domy = np.abs(median_shift_left - 2*median_blur_image + median_shift_right)
    
    return domx, domy
    
    
domx, domy = dom(Im)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5))
showImage(image/np.max(image), ax1, fig, "Grayscale Image", "gray")
showImage(domx, ax2, fig, "$DOM_x$", "gray")
showImage(domy, ax3, fig, "$DOM_y$", "gray") 


#DoM and Edge Width

def contrast(Im):
    Cx = np.abs(Im - np.pad(Im, ((1,0), (0,0)), 'constant')[:-1, :])
    Cy = np.abs(Im - np.pad(Im, ((0,0), (1,0)), 'constant')[:, :-1])
    return Cx, Cy

Cx, Cy = contrast(Im)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 5))
showImage(image/np.max(image), ax1, fig, "Grayscale Image", "gray")
showImage(Cx, ax2, fig, "$C_x$", "gray")
showImage(Cy, ax3, fig, "$C_y$", "gray")

#Image Smoothing

def smoothenImage(gray, transpose = False, epsilon = 1e-8):
    fil = np.array([0.5, 0, -0.5])
    
    if transpose:
        gray = gray.T
        
    gray_smoothed = np.array([np.convolve(gray[i], fil, mode="same") for i in range(gray.shape[0])])
    
    if transpose:
        gray_smoothed = gray_smoothed.T

#     gray_smoothed = convolve1d(gray, fil, mode="constant", axis=transpose) 
    gray_smoothed = np.abs(gray_smoothed)/(np.max(gray_smoothed) + epsilon)
    return gray_smoothed

#Sharpness Measure

def sharpness_matrix(Im, edgex, edgey, width=2, debug=False):
    
    dodx, dody = dom(Im) #compute_dod(Im)
    Cx, Cy = contrast(Im)
    
    Cx = np.multiply(Cx, edgex)
    Cy = np.multiply(Cy, edgey)
    
    Sx = np.zeros(dodx.shape)
    Sy = np.zeros(dody.shape)
    
    for i in range(width, dodx.shape[0]-width):
        num = np.abs(dodx[i-width:i+width, :]).sum(axis=0)
        dn = Cx[i-width:i+width, :].sum(axis=0)
        Sx[i] = [(num[k]/dn[k] if dn[k] > 1e-3 else 0) for k in range(Sx.shape[1])]
    
    for j in range(width, dody.shape[1]-width):
        num = np.abs(dody[:, j-width: j+width]).sum(axis=1)
        dn = Cy[:, j-width:j+width].sum(axis=1)
        Sy[:, j] = [(num[k]/dn[k] if dn[k] > 1e-3 else 0) for k in range(Sy.shape[0])]
        
    if debug:
        print(f"dodx {dodx.shape}: {[(i,round(np.quantile(dodx, i/100), 2)) for i in range(0, 101, 25)]}")
        print(f"dody {dody.shape}: {[(i,round(np.quantile(dody, i/100), 2)) for i in range(0, 101, 25)]}")
        print(f"Cx {Cx.shape}: {[(i,round(np.quantile(Cx, i/100),2)) for i in range(50, 101, 10)]}")
        print(f"Cy {Cy.shape}: {[(i,round(np.quantile(Cy, i/100),2)) for i in range(50, 101, 10)]}")
        print(f"Sx {Sx.shape}: {[(i,round(np.quantile(Sx, i/100),2)) for i in range(50, 101, 10)]}")
        print(f"Sy {Sy.shape}: {[(i,round(np.quantile(Sy, i/100),2)) for i in range(50, 101, 10)]}")
        
    return Sx, Sy, dodx, dody, Cx, Cy

# Sharpness Calculation

def compute_sharpness(Sx, Sy, edgex, edgey, sharpness_threshold=1, debug=False):

    epsilon = 1e-8

    Sx = np.multiply(Sx, edgex)
    n_sharpx = np.sum(Sx >= sharpness_threshold)
    
    Sy = np.multiply(Sy, edgey)
    n_sharpy = np.sum(Sy >= sharpness_threshold)

    n_edgex = np.sum(edgex)
    n_edgey = np.sum(edgey)
    
    Rx = n_sharpx/(n_edgex + epsilon)
    Ry = n_sharpy/(n_edgey + epsilon)

    S = round(np.sqrt(Rx**2 + Ry**2)/np.sqrt(2), 2) * 100
    
    if debug:
        print(f"Sharpness: {S}")
        print(f"Rx: {Rx}, Ry: {Ry}")
        print(f"Sharpx: {n_sharpx}, Sharpy: {n_sharpy}, Edges: {n_edgex, n_edgey}")
    return S

width = 1 # 1
edge_threshold = 0.0001
sharpness_threshold = 0.75

r = 2
c = 3

fig, axes = plt.subplots(r,c, figsize=(16, 10))
ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

image, Im = load(imagePath)
showImage(image/255, ax1, fig, "Original Image", "gray")

smoothx = smoothenImage(image, transpose=True)
smoothy = smoothenImage(image)

edgex = smoothx > edge_threshold
edgey = smoothy > edge_threshold

showImage(Im, ax2, fig, f"Median Filtered Image")


Sx, Sy, dodx, dody, Cx, Cy = sharpness_matrix(Im, edgex, edgey, width=width, debug=True)

show = ["x", "y"][0]

if show == "x":

    showImage(smoothx, ax3, fig, "Smooth X")
    showImage(dodx, ax4, fig, "DOMx")
    showImage(Cx, ax5, fig, "Cx")
    Sx = np.clip(Sx, 0, 10)
    showImage(Sx, ax6, fig, "Sx")
    
elif show == "y":
    showImage(smoothy, ax3, fig, "Smooth Y")
    showImage(dody, ax4, fig, "DOMy")
    showImage(Cy, ax5, fig, "Cy")
    Sy = np.clip(Sy, 0, 10)
    showImage(Sy, ax6, fig, "Sy")

plt.show()

score = compute_sharpness(Sx, Sy, edgex, edgey, sharpness_threshold, True)


"""
fig2, (ax1 , ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

x1, x2 = 40, 80 
y1, y2 = 170, 200

showImage(image/np.max(image), ax1, fig2, "Original Image")

cropped_image = image[x1:x2, y1:y2]
showImage(cropped_image/np.max(cropped_image), ax2, fig2, "Cropped Original Image")
show = ['x', 'y'][0]

if show == 'x':
    Sx = np.clip(Sx, 0, 10)
    showImage(Sx[x1:x2, y1:y2], ax3, fig2, "Sx")
else:
    Sy = np.clip(Sy, 0, 10)
    showImage(Sy[x1:x2, y1:y2], ax3, fig2, "Sy")"""

