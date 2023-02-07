#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 17:39:16 2022

@author: venkat
"""
import numpy as np
import cv2
from skimage import color, data, restoration
camera = color.rgb2gray(data.astronaut())
from scipy.signal import convolve2d
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import imquality.brisque as brisque
from niqe import niqe
from piqe import piqe
import time


all_results = []

for win_length in np.arange(3,21,2):
    
    psf = np.ones((3, 3)) / 9
    
    #camera1 = convolve2d(camera, psf, 'same')
    start_time = time.time() 
    
    camera1=cv2.GaussianBlur(camera, (win_length,win_length), 0)
    
    np.random.seed(0)
    
    camera_noisy =camera1+0.01 * camera1.std() * np.random.standard_normal(camera1.shape)
    
    deconvolved = restoration.richardson_lucy(camera_noisy, psf, 2)
    
    
    cv2.imshow('', deconvolved )
    
    MSE = mean_squared_error(camera, deconvolved)
    
    PSNR = peak_signal_noise_ratio(camera, deconvolved)
    
    SSIM=ssim(camera, deconvolved)
    
    print('MSE: ', MSE)
    print('PSNR: ', PSNR)
    print('SSIM:', SSIM )
    
    brwisq_scr = brisque.score(deconvolved)
     
    print('brisque score=', brwisq_scr)
     
    niqe_score= niqe(deconvolved)
         
    print('niqe_score=', niqe_score)
         
    piqe_score= piqe(deconvolved)
         
    print('piqe_score=', piqe_score)
    
    result = [MSE,PSNR,SSIM, brwisq_scr, niqe_score,piqe_score, time.time() - start_time]
    
    all_results.append(result)
    
all_results = np.float32(all_results)
np.savetxt('/home/venkat/Richardson_Lucy/test4.csv',all_results,'%10.5f',header='MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score, Iteration_time', delimiter=',')