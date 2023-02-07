#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:03:40 2022

@author: venkat
"""



import cv2 
import numpy as np



from niqe import niqe
from piqe import piqe
import imquality.brisque as brisque
#%% load the input image and convert it grey scale image


if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''
    
    im = cv2.imread("/home/venkat/Data Set/1_1.jpg")
    
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#%%    Genarate Gaussian Noise and add to the image
    
    gauss=np.random.normal(0,1, im.shape)
    
    #Add noise to the image
    
    noisy_img=im+10*gauss
    
#%%blur the image
    
    
    blurred=cv2.GaussianBlur(im, (5,5), 0)
    
#%%  Calculate no reference image performance metrics
  
    print('brisque score=', brisque.score(noisy_img))

    
    niqe_score= niqe(noisy_img)
    
    print('niqe_score=', niqe_score)
    
    piqe_score= piqe(noisy_img)
    
    print('piqe_score=', piqe_score)