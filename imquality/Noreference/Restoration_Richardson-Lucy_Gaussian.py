#%% Import all necessary packages
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
img_dir = "/home/venkat/Data_Set"  
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
    
    for win_length in np.arange(15,17,2):
    #for win_length in [7]:
        
               
        start_time = time.time()  
        
        print('\n\n Processing : win : %d' %(win_length))
#%% Blur the image        
        
        #blurred = conv2(img1, (win_length, win_length), 'same')
        
        
        blurred=cv2.GaussianBlur(img1, (win_length,win_length), 2)
    
        #blurred1=np.uint8(blurred)
    
    #noisy_image1 = noisy_image.astype(np.uint8)
        #Gaussian_blur=img1-blurred
        
#%% Add noise to the blurred image    
        """for var in np.arange(0.9,1,0.1):
        #for var in [0.5]:

            np.random.seed(0)
                
            gauss=np.random.normal(0,var, blurred.shape)
            
            #blurred += (rng.poisson(lam=2, size=blurred.shape) - 10) / 255
            
            # #Add noise to the image
            noisy_img=blurred+gauss
            
            #astro_blur=astro_blur.astype(np.uint8)
            # Blur the Image
            print('\n\n noise_variance : var : %f' %(var))"""
            #astro_blur=cv2.blur(astro,(3,3))
    #%% Generate deblurred image using Richardson_lucy algorithm
            
            # Restore Image using Richardson-Lucy algorithm
            
            #dummy_psf = np.zeros((5,5))
            
            #dummy_psf[dummy_psf.shape[0]//2,dummy_psf.shape[1]//2]=1
            
        #dummy_psf = np.ones((9, 9)) /81
           
        """dummy_psf=[[0.00624107, 0.12535524, 0.34075086, 0.34075086, 0.12535524],
         [0.00636511, 0.12784659, 0.34752306, 0.34752306, 0.12784659],
         [0.00649043, 0.13036379, 0.35436551, 0.35436551, 0.13036379],
         [0.00661702, 0.13290642, 0.3612771,  0.3612771,  0.13290642],
         [0.00674486, 0.13547405, 0.36825665, 0.36825665, 0.13547405]]"""
        
        """dummy_psf=[[6.24140166e-03, 1.25361904e-01,3.40768985e-01, 3.40768985e-01,
          1.25361904e-01],
         [1.17476411e-02, 2.35957680e-01, 6.41399473e-01, 6.41399473e-01,
          2.35957680e-01],
         [6.05451509e-04, 1.21608186e-02, 3.30565323e-02, 3.30565323e-02,
          1.21608186e-02],
         [1.74682137e-02, 3.50858452e-01, 9.53732154e-01, 9.53732154e-01,
          3.50858452e-01],
         [1.11424191e-02, 2.23801471e-01, 6.08355471e-01, 6.08355471e-01,
          2.23801471e-01]]"""
         
        """dummy_psf=[[0.00624107, 0.00624107, 0.12535524, 0.34075086, 0.34075086, 0.12535524,
          0.12535524],
         [0.00632362, 0.00632362, 0.12701325, 0.3452578,  0.3452578,  0.12701325,
          0.12701325],
         [0.00640674, 0.00640674, 0.1286828,  0.34979612, 0.34979612, 0.1286828,
          0.1286828 ],
         [0.00649043, 0.00649043, 0.13036379, 0.35436551, 0.35436551, 0.13036379,
          0.13036379],
         [0.00657468, 0.00657468, 0.13205608, 0.35896563, 0.35896563, 0.13205608,
          0.13205608],
         [0.0066595,  0.0066595,  0.13375954, 0.36359613, 0.36359613, 0.13375954,
          0.13375954],
         [0.00674486, 0.00674486, 0.13547405, 0.36825665, 0.36825665, 0.13547405,
          0.13547405]]"""
        
        
        dummy_psf=[[6.24140166e-03, 6.24140166e-03, 1.25361904e-01, 3.40768985e-01,
          3.40768985e-01, 1.25361904e-01, 1.25361904e-01],
         [8.98910096e-03, 8.98910096e-03, 1.80550919e-01, 4.90788283e-01,
          4.90788283e-01, 1.80550919e-01, 1.80550919e-01],
         [1.35604371e-02, 1.35604371e-02, 2.72368659e-01, 7.40374777e-01,
          7.40374777e-01, 2.72368659e-01, 2.72368659e-01],
         [4.27369794e-04, 4.27369794e-04, 8.58395178e-03, 2.33336001e-02,
          2.33336001e-02, 8.58395178e-03, 8.58395178e-03],
         [1.74682137e-02, 1.74682137e-02, 3.50858452e-01, 9.53732154e-01,
          9.53732154e-01, 3.50858452e-01, 3.50858452e-01],
         [1.47696158e-02, 1.47696158e-02, 2.96655664e-01, 8.06393701e-01,
          8.06393701e-01, 2.96655664e-01 ,2.96655664e-01],
         [6.74485576e-03, 6.74485576e-03, 1.35474049e-01, 3.68256647e-01,
          3.68256647e-01, 1.35474049e-01, 1.35474049e-01]]
        
        """dummy_psf=[[6.24140166e-03 6.24140166e-03 1.25361904e-01 3.40768985e-01
          3.40768985e-01 1.25361904e-01 1.25361904e-01]
         [8.98910096e-03 8.98910096e-03 1.80550919e-01 4.90788283e-01
          4.90788283e-01 1.80550919e-01 1.80550919e-01]
         [1.35604371e-02 1.35604371e-02 2.72368659e-01 7.40374777e-01
          7.40374777e-01 2.72368659e-01 2.72368659e-01]
         [4.27369794e-04 4.27369794e-04 8.58395178e-03 2.33336001e-02
          2.33336001e-02 8.58395178e-03 8.58395178e-03]
         [1.74682137e-02 1.74682137e-02 3.50858452e-01 9.53732154e-01
          9.53732154e-01 3.50858452e-01 3.50858452e-01]
         [1.47696158e-02 1.47696158e-02 2.96655664e-01 8.06393701e-01
          8.06393701e-01 2.96655664e-01 2.96655664e-01]
         [6.74485576e-03 6.74485576e-03 1.35474049e-01 3.68256647e-01
          3.68256647e-01 1.35474049e-01 1.35474049e-01]]"""
           
        """
        dummy_psf=[[0.00624107, 0.00624107, 0.12535524, 0.12535524, 0.34075086,
                0.34075086, 0.34075086, 0.12535524, 0.12535524],
               [0.00630293, 0.00630293, 0.12659766, 0.12659766, 0.34412811,
                0.34412811, 0.34412811, 0.12659766, 0.12659766],
               [0.00636511, 0.00636511, 0.12784659, 0.12784659, 0.34752306,
                0.34752306, 0.34752306, 0.12784659, 0.12784659],
               [0.00642761, 0.00642761, 0.12910198, 0.12910198, 0.35093557,
                0.35093557, 0.35093557, 0.12910198, 0.12910198],
               [0.00649043, 0.00649043, 0.13036379, 0.13036379, 0.35436551,
                0.35436551, 0.35436551, 0.13036379, 0.13036379],
               [0.00655357, 0.00655357, 0.13163195, 0.13163195, 0.35781274,
                0.35781274, 0.35781274, 0.13163195, 0.13163195],
               [0.00661702, 0.00661702, 0.13290642, 0.13290642, 0.3612771 ,
                0.3612771 , 0.3612771 , 0.13290642, 0.13290642],
               [0.00668078, 0.00668078, 0.13418714, 0.13418714, 0.36475846,
                0.36475846, 0.36475846, 0.13418714, 0.13418714],
               [0.00674486, 0.00674486, 0.13547405, 0.13547405, 0.36825665,
                0.36825665, 0.36825665, 0.13547405, 0.13547405]]"""
        
        dummy_psf=[[1.20917495e-03, 6.60187155e-02, 2.42869281e-02],
         [1.74149915e-03, 9.50826321e-02, 3.49789456e-02],
         [2.62712476e-03, 1.43436152e-01, 5.27672113e-02],
         [8.27962817e-05, 4.52052381e-03, 1.66300777e-03],
         [3.38419599e-03, 1.84770841e-01, 6.79733936e-02],
         [2.86138443e-03, 1.56226296e-01, 5.74724425e-02],
         [1.30671139e-03 ,7.13440245e-02, 2.62459999e-02]]


        dummy_psf = np.array(dummy_psf, dtype=np.float64)
            
            #dummy_psf=np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]]) # Prewitt
            
            #dummy_psf=np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Laplacian
            
            #out = bm3d.bm3d(noisy_img, sigma_psd=.1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
            
            #LL, (LH, HL, HH) = pywt.dwt2(noisy_img, 'haar')  # Wavelet
            
           # out = pywt.idwt2((LL, (LH, HL, None)), 'haar')
           
            #out=wiener(noisy_img, (7, 7))   # Weiner filtering
            
            #denoisy =restoration.unsupervised_wiener(noisy_img, dummy_psf) # Unsupervised weiner filtering
            
            
            
        deconvolved_RL = restoration.richardson_lucy(blurred, dummy_psf , num_iter=10)
            
            
            
            #deconvolved_RL =restoration.unsupervised_wiener(blurred, dummy_psf)
            
            #deconvolved_RL=img_as_ubyte(deconvolved_RL)
            
    #%% Display Original, Noisy_image, Denoised images        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
                
        plt.gray()
                
        for a in (ax[0], ax[1], ax[2]):
          a.axis('off')
                
        ax[0].imshow(img1)
        ax[0].set_title('Original Data')
                
        ax[1].imshow(blurred)
        ax[1].set_title('Blur, win_length=%d' % (win_length))
                
        ax[2].imshow(deconvolved_RL, vmin=blurred.min(), vmax=blurred.max())
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
            
    
            
    brwisq_scr = brisque.score(deconvolved_RL)
            
    print('brisque score=', brwisq_scr)
            
        
                
            #print('Error Msg')
            
    niqe_score= niqe(deconvolved_RL)
                
    print('niqe_score=', niqe_score)
                
    piqe_score= piqe(deconvolved_RL)
                
    print('piqe_score=', piqe_score)
                
    print('Iteration time', time.time() - start_time) 
                
    result = [i,win_length,MSE,PSNR,SSIM,brwisq_scr, niqe_score,piqe_score, time.time() - start_time]
            
            #result = [i,win_length,var, MSE,PSNR,SSIM, niqe_score,piqe_score, time.time() - start_time]
                
                               
    """if win_length==1:
                    
                    calibration_fact = result
                    
                    calibration_fact[calibration_fact.index(0)]=1
                    
                    
                    calibration_fact[calibration_fact.index(np.inf)]=1
                    #calibration_fact[calibration_fact==0]=1
                    
                    #calibration_fact[[np.isinf(calibration_fact)]==True]=1
                    
                result = [x/y for x, y in zip(result,calibration_fact)]"""
                
    all_results.append(result)
         
        
all_results = np.float32(all_results)

np.savetxt('/home/venkat/RL/dummy_psf_5X5.csv',all_results,'%10.5f',header='i, win_length, MSE,PSNR,SSIM,brwisq_scr,niqe_score,piqe_score, Iteration_time', delimiter=',')