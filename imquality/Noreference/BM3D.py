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
#from skimage.util import random_noise
import cv2
import imquality.brisque as brisque
from niqe import niqe
from piqe import piqe

#import unittest

all_results = []

for win_lenght in np.arange(3,23,2):
    
    print('\n\n Processing : win : %d' %(win_lenght))

    #%% Read the image and convert it into Grey Scale Image
    
    astro = cv2.imread('/home/venkat/Data Set/img1.png')
    astro1=cv2.cvtColor(astro, cv2.COLOR_BGR2GRAY)
    
    #%%Generate Gaussian Noise and Add to the image
    np.random.seed(0)
    gauss=np.random.normal(0,1, astro1.shape)
    
    #Add noise to the image
    noisy_img=astro1+10*gauss
    
    
    #%% Blur the image and add to the image
    blurred=cv2.GaussianBlur(astro1, (win_lenght,win_lenght), 0)
    
    blurred1=np.uint8(blurred)
    
    #noisy_image1 = noisy_image.astype(np.uint8)
    
    #%% Generate denoised image using BM3D algorithm
    
    out = bm3d.bm3d(blurred1, sigma_psd=1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
    
    #%% Display Original, Noisy_image, Denoised images
    plt = 0
    if plt:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
        plt.gray()
        
        for a in (ax[0], ax[1], ax[2]):
               a.axis('off')
        
        ax[0].imshow(astro1)
        ax[0].set_title('Original Data')
        
        ax[1].imshow(noisy_img)
        ax[1].set_title('Blurred_img')
        
        ax[2].imshow(out)
        ax[2].set_title('Deblurred_img')
        
        fig.subplots_adjust(wspace=0.1, hspace=0.5,
                            top=0.9, bottom=0.05, left=0, right=1)
        plt.show()
        
    
    #%% calculate performance metrics MSE, PSNR, SSIM
    
    MSE = mean_squared_error(astro1, out)
    PSNR = peak_signal_noise_ratio(astro1, out, data_range=astro1.max() - astro1.min())
    SSIM=ssim(astro1, out)
    print('MSE: ', MSE)
    print('PSNR: ', PSNR)
    print('SSIM:', SSIM )
    
    #%%  Calculate no reference image performance metrics
      
    brwisq_scr = brisque.score(out)
    print('brisque score=', brwisq_scr)
    
        
    niqe_score= niqe(out)
        
    print('niqe_score=', niqe_score)
        
    piqe_score= piqe(out)
        
    print('piqe_score=', piqe_score)
        
    
    #%% 
    result = [win_lenght, MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score]
    
    all_results.append(result)
    
    
all_results = np.float32(all_results)
np.savetxt('/home/venkat/test.csv',all_results,'%10.5f',header='win_lenght, MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score',delimiter=',')

