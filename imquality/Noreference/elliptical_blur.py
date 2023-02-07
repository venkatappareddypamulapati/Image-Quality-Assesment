##%% Import all necessary packages
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
import cv2
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
import os
import glob
import time
import imquality.brisque as brisque
from niqe import niqe
from piqe import piqe
import bm3d
import pywt
from pywt import wavedec
from scipy.signal import wiener

rng = np.random.default_rng()
#%%#%% get the path/directory
img_dir = "/home/venkat/Data Set"  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
files = files[0:1]

all_results = []

for i,f1 in enumerate(files): 
    
    img = cv2.imread(f1) 
    img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1=img_as_float(img1)
    # astro1 = astro1[:200,:200]
    print('\n\n Deblurring of Img: %d' %(i))        
#%% Define win_length for psf and start the clock
    print('\n\n')
    
    for win_length in np.arange(13,15,2):
    #for win_length in [7]:
        
               
        start_time = time.time()  
        
        print('\n\n Processing : win : %d' %(win_length))
#%% Blur the image        
        
        #blurred = conv2(img1, (win_length, win_length), 'same')
        
        
        #blurred=cv2.GaussianBlur(img1, (win_length,win_length), 0)
        
        psf=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(win_length,win_length)) # elliptical kernel
        
        psf=psf.astype(np.float64)
        
        #psf=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)) # cross kernel
        
       # psf[psf.shape[0]//2, psf.shape[1]//2]= psf[psf.shape[0]//2, psf.shape[1]//2]*4
        
        blurred = conv2(img1, psf, 'same')
        
        cv2.imwrite('/home/venkat/Desktop/images/elli.png', blurred)
    
        #blurred1=np.uint8(blurred)
    
        #noisy_image1 = noisy_image.astype(np.uint8)
        #Gaussian_blur=img1-blurred
        
#%% Add noise to the blurred image    
        #for var in np.arange(0.4,0.5,0.1):
        #for var in [0.5]:

            #np.random.seed(0)
                
            #gauss=np.random.normal(0,var, blurred.shape)
            
           # blurred += (rng.poisson(lam=2, size=blurred.shape) - 10) / 255
            
            # #Add noise to the image
           # noisy_img=blurred+gauss
            
            #astro_blur=astro_blur.astype(np.uint8)
            # Blur the Image
            #print('\n\n noise_variance : var : %f' %(var))
            #astro_blur=cv2.blur(astro,(3,3))
    #%% Generate deblurred image using Richardson_lucy algorithm
            
            # Restore Image using Richardson-Lucy algorithm
            
            #dummy_psf = np.zeros((5,5))
            
            #dummy_psf[dummy_psf.shape[0]//2,dummy_psf.shape[1]//2]=1
            
            #dummy_psf = np.ones((3, 3)) /9
            
        #dummy_psf=np.array([[1, 1, 1,1 ,1,1,1],
            #     [1, 1, 1, 1, 1,1,1],
            #     [1, 1, 1, 1, 1,1,1]])
            
            #dummy_psf=(1/sum(map(sum, dummy_psf)))* dummy_psf
            
            #dummy_psf=np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]]) # Prewitt
            
            #dummy_psf=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Laplacian
            
            #out = bm3d.bm3d(blurred, sigma_psd=.01, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
            
            #LL, (LH, HL, HH) = pywt.dwt2(noisy_img, 'db2')  # Wavelet
            
            #out = pywt.idwt2((LL, (LH, HL, None)), 'db2')
           
            #out=wiener(noisy_img, (9,9))   # Weiner filtering
            
            #deconvolved_RL=restoration.unsupervised_wiener(noisy_img, dummy_psf) # Unsupervised weiner filtering
            
        dummy_psf=[[0.00767001, 0.00767001, 0.15405631, 0.15405631, 0.41876847,
                0.41876847, 0.41876847, 0.15405631, 0.15405631],
               [0.00649672, 0.00649672, 0.13049021, 0.13049021, 0.35470915,
                0.35470915, 0.35470915, 0.13049021, 0.13049021],
               [0.0054238 , 0.0054238 , 0.10893991, 0.10893991, 0.29612938,
                0.29612938, 0.29612938, 0.10893991, 0.10893991],
               [0.00446296, 0.00446296, 0.08964102, 0.08964102, 0.24366955,
                0.24366955, 0.24366955, 0.08964102, 0.08964102],
               [0.00361954, 0.00361954, 0.07270047, 0.07270047, 0.19762037,
                0.19762037, 0.19762037, 0.07270047, 0.07270047],
               [0.00289331, 0.00289331, 0.05811368, 0.05811368, 0.15796935,
                0.15796935, 0.15796935, 0.05811368, 0.05811368],
               [0.00227954, 0.00227954, 0.04578573, 0.04578573, 0.12445852,
                0.12445852, 0.12445852, 0.04578573, 0.04578573],
               [0.00177015, 0.00177015, 0.03555434, 0.03555434, 0.09664672,
                0.09664672, 0.09664672, 0.03555434, 0.03555434],
               [0.00135482, 0.00135482, 0.02721233, 0.02721233, 0.07397079,
                0.07397079, 0.07397079, 0.02721233, 0.02721233]]
    
        """dummy_psf=[[1.49206225e-02, 1.01492062e+00, 9.01492062e+00],
              [9.45270063e-03, 1.00945270e+00, 9.00945270e+00],
              [5.22729000e-03, 1.00522729e+00, 9.00522729e+00]]"""
        
        dummy_psf = np.array(dummy_psf, dtype=np.float64)
            
        deconvolved_RL = restoration.richardson_lucy(blurred, dummy_psf , num_iter=10)
            
            
            
            #deconvolved_RL =restoration.unsupervised_wiener(blurred, dummy_psf)
            
    #        deconvolved_RL=img_as_ubyte(deconvolved_RL)
            
    #%% Display Original, Noisy_image, Denoised images        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
            
        plt.gray()
            
        for a in (ax[0], ax[1], ax[2]):
                   a.axis('off')
            
        ax[0].imshow(img1, cmap='gray')
        ax[0].set_title('Original Data')
            
        ax[1].imshow(blurred, cmap='gray')
        ax[1].set_title('Blur + Noisy data, win_length=%d' % (win_length))
            
        ax[2].imshow(deconvolved_RL, cmap='gray')
        ax[2].set_title('Restoration using\nRichardson-Lucy')
            
            
        fig.subplots_adjust(wspace=0.02, hspace=0.2,
                                top=0.9, bottom=0.05, left=0, right=1)
        plt.show()
        
     #%% calculate performance metrics MSE, PSNR, SSIM        
        MSE = mean_squared_error(img1, deconvolved_RL)
        PSNR = peak_signal_noise_ratio(img1, deconvolved_RL)
        SSIM=ssim(img1, deconvolved_RL)
        print('MSE: ', MSE)
        print('PSNR: ', PSNR)
        print('SSIM:', SSIM )
            
    
            
        #brwisq_scr = brisque.score(deconvolved_RL)
            
        #print('brisque score=', brwisq_scr)
        
        
                
            #print('Error Msg')
            
        niqe_score= niqe(deconvolved_RL)
                
        print('niqe_score=', niqe_score)
                
        piqe_score= piqe(deconvolved_RL)
                
        print('piqe_score=', piqe_score)
                
        print('Iteration time', time.time() - start_time) 
                
        #result = [i,win_length,MSE,PSNR,SSIM,brwisq_scr, niqe_score,piqe_score, time.time() - start_time]
            
        result = [i,win_length, MSE,PSNR,SSIM, niqe_score,piqe_score, time.time() - start_time]
                
               
                    
        """if win_length==1:
                    
                    calibration_fact = result
                    
                    calibration_fact[calibration_fact.index(0)]=1
                    
                    
                    calibration_fact[calibration_fact.index(np.inf)]=1
                    #calibration_fact[calibration_fact==0]=1
                    
                    #calibration_fact[[np.isinf(calibration_fact)]==True]=1
                    
                result = [x/y for x, y in zip(result,calibration_fact)]"""
                
    all_results.append(result)
         
        
all_results = np.float32(all_results)

np.savetxt('/home/venkat/RL/dummy_psf_5X5.csv',all_results,'%10.5f',header='i, win_length,  MSE,PSNR,SSIM,niqe_score,piqe_score, Iteration_time', delimiter=',')