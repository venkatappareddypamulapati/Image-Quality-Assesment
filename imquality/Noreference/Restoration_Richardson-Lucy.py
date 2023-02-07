# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 14:41:45 2022

@author: venkat
"""

#%% Import all necessary packages
import bm3d
import numpy as np
#from collections import deque
#import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import wiener
#import argparse
#from skimage.util import random_noise
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
#from skimage import exposure
#import time
#from skimage.util import random_noise
from scipy.signal import convolve2d as conv2
import cv2
import imquality.brisque as brisque
from niqe import niqe
from piqe import piqe
from skimage import color, data, restoration
import os
import glob
import time
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

#%% get the path/directory
img_dir = "/home/venkat/Data Set"  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
files = files[0:1]

all_results = []

for i,f1 in enumerate(files): 
    img = cv2.imread(f1) 
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # astro1 = astro1[:200,:200]
    print('\n\n Deblurring of Img: %d' %(i))        
#%% Define win_length
    print('\n\n')

    for win_length in np.arange(7,9,2):
        
        
        start_time = time.time()  
        
        print('\n\n Processing : win : %d' %(win_length))

   
    #%% Blur the image and add to the image
    
        psf=np.ones((7, 7)) / 49
        
        blurred=conv2(img1, psf, 'same')
        
       
        #blurred=cv2.GaussianBlur(img1, (win_lenght,win_lenght), 1)
    
        #blurred1=np.uint8(blurred)
    
    #noisy_image1 = noisy_image.astype(np.uint8)
        #Gaussian_blur=img1-blurred
    
    #%%Generate Gaussian Noise and Add to the image
        np.random.seed(0)
        
        gauss=np.random.normal(0,1, blurred.shape)
    
    # #Add noise to the image
        noisy_img=blurred+gauss
    
  
    
    #%% Generate denoised image using BM3D algorithm
    
        #out = bm3d.bm3d(blurred1, sigma_psd=1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        #out = blurred
    #%% Generate deblurred image using Richardson_lucy algorithm
    
        deconvolved_RL_20 = restoration.richardson_lucy(blurred, psf, num_iter=100)
        
        deconvolved_RL_20=255*np.uint8(deconvolved_RL_20)
        
        #deconvolved_RL_20=img_as_ubyte(deconvolved_RL_20)
        
        #deconvolved_RL_20=np.uint8(deconvolved_RL_20)
      
 #%% Display Original, Noisy_image, Denoised images
        #plt = 0
        #if plt:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
        plt.gray()

        for a in (ax[0], ax[1], ax[2]):
               a.axis('off')

        ax[0].imshow(img1)
        ax[0].set_title('Original Data')

        ax[1].imshow(blurred)
        ax[1].set_title('Blur + Noisy data')

        ax[2].imshow(deconvolved_RL_20, vmin=blurred.min(), vmax=blurred.max())
        ax[2].set_title('Restoration using\nRichardson-Lucy')
        
    
    #%% calculate performance metrics MSE, PSNR, SSIM
    
        MSE = mean_squared_error(img1, deconvolved_RL_20)
        PSNR = peak_signal_noise_ratio(img1, deconvolved_RL_20)
        SSIM=ssim(img1, deconvolved_RL_20)
        print('MSE: ', MSE)
        print('PSNR: ', PSNR)
        print('SSIM:', SSIM )
    
    #%%  Calculate no reference image performance metrics
      
        
        brwisq_scr = brisque.score(deconvolved_RL_20)
        print('brisque score=', brwisq_scr)
    
        niqe_score= niqe(deconvolved_RL_20)
        
        print('niqe_score=', niqe_score)
        
        piqe_score= piqe(  deconvolved_RL_20)
        
        print('piqe_score=', piqe_score)
        
        print('Iteration time', time.time() - start_time)   
    #%% Storing all the results
    
        result = [i,win_length, MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score, time.time() - start_time]
        
        """if win_length==1:
            
            calibration_fact = result
            
            calibration_fact[calibration_fact.index(0)]=1
            
            
            calibration_fact[calibration_fact.index(np.inf)]=1
            #calibration_fact[calibration_fact==0]=1
            
            #calibration_fact[[np.isinf(calibration_fact)]==True]=1
            
        result = [x/y for x, y in zip(result,calibration_fact)]"""
        
        all_results.append(result)
        
all_results = np.float32(all_results)
np.savetxt('/home/venkat/Richardson_Lucy/test1.csv',all_results,'%10.5f',header='id, win_lenght, MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score, Iteration_time', delimiter=',')

#plt.imshow(psf)