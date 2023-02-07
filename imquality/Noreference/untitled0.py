#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:23:19 2022

@author: venkat
"""

import numpy as np
from skimage.io import imread, imsave
from pyclesperanto_prototype import imshow
import pyclesperanto_prototype as cle
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#%%

bead_image = cv2.imread('/home/venkat/Data Set/img1.png')
bead_image.shape

#%%
imshow(cle.maximum_x_projection(bead_image), colorbar=True)
imshow(cle.maximum_y_projection(bead_image), colorbar=True)
imshow(cle.maximum_z_projection(bead_image), colorbar=True)


label_image = cle.voronoi_otsu_labeling(bead_image)
imshow(label_image, labels=True)


stats = cle.statistics_of_labelled_pixels(bead_image, label_image)

df = pd.DataFrame(stats)
df[["mass_center_x", "mass_center_y", "mass_center_z"]]

