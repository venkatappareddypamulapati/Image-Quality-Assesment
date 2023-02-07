from PyEMD.EMD2d import EMD2D  #, BEMD
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_float, random_noise
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import imquality.brisque as brisque
from niqe import niqe
from piqe import piqe
#%%

img=cv2.imread('/home/venkat/Data Set/img1.png')[:500, :500,:]
    
img=img_as_float(img)
    
img=rgb2gray(img)

out_img=np.zeros(img.shape)
emd2d = EMD2D()  

for i in np.arange(0,10,1):
    
    
    print(i)  
  
        
    rand=random_noise(img, mode='gaussian', seed=None, clip=True, mean=0, var=1)
        
    noisy_img=img+rand
        
    
        
    IMFs_2D = emd2d(noisy_img, max_imf=-1)
        
    IMFs_2D_sum=IMFs_2D[0]+IMFs_2D[1]
    
   # print(np.abs(IMFs_2D_sum-noisy_img).sum())
        
    out_img+=IMFs_2D_sum/100
        
IMF_0=out_img

#%%

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(noisy_img, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(IMF_0, cmap='gray')

#%%
"""brwisq_scr = brisque.score(IMF_0)
 
print('brisque score=', brwisq_scr)
 
 
niqe_score= niqe(IMF_0)
     
print('niqe_score=', niqe_score)
     
piqe_score= piqe(IMF_0)
     
print('piqe_score=', piqe_score)"""

psnr_emd=psnr(img, IMF_0)
print(psnr_emd)
ssim_emd=ssim(img, IMF_0)
print(ssim_emd)
mse_emd=mse(img, IMF_0)
print(mse_emd)
