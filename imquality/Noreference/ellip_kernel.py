
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:49:37 2022

@author: venkat
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from mpl_toolkits import mplot3d
plt.style.use('seaborn-poster')
 
# sigma(standard deviation) and muu(mean) are the parameters of gaussian
 
def ellip_filter(kernel_size,a,b,p,q):
 
     
   
    dst= (((x-p) // a) ** 2)+ (((y-q) / b)** 2)-1
 

 
    return dst/dst.sum()

kernel_size=5

a=5
b=7
p=0
q=0

#x, y = np.meshgrid(np.linspace(0, 12, kernel_size),np.linspace(0.7329, 0.4338, kernel_size))
  
x=np.linspace(0,12, kernel_size)
  
#y=[0.7329,-1.7906,-5.8205,-11.8142,-13.9608,-15.8853,-12.1752,-10.5079,-6.7336,-4.1944,-0.3426,2.2528,0.4337]

y=[0.7329, -11.8142, -12.1752, -4.1944, 0.4337]  

#y=np.array(y)
  
x,y=np.meshgrid(x,y)

ellipse = ellip_filter(kernel_size, a,b,p,q)

print("ellipse filter of{} X {} :".format(kernel_size,kernel_size))
print(ellipse)

ellipse=np.array(ellipse)
"""fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

surf = ax.plot_surface(ellipse,x,y, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()"""
img=cv2.imread('/home/venkat/Downloads/Siemens.png', 0)
blur=cv2.filter2D(img, -1, ellipse)
cv2.imwrite('/home/venkat/Downloads/blurred.png', blur)

