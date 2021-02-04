import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage

import glob

# We have stored some random apparel color images and 
# trying to convert all the images in gray scale images


# First download the images from my github 
# replace the url below with your local repo path

#file = 'https://github.com/azhar2ds/DataSets/tree/master/Apparel_Random_Dataset/*.*'



path = 'color_dataset/*.*'
img = glob.glob(path)
lst=[]
img_num = 1

for x in img:
    a = cv2.imread(x)
    gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_dataset/' +str(img_num)+ '.jpg', gray)
    img_num += 1
    cv2.imshow('gray', gray)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
 
# Now lets check out how our gray scale images looks like.


p = glob.glob('gray_dataset/*.*')
gr_lst =[]
for k in p:
    a =cv2.imread(k)
    gr_lst.append(a)
    for c in range(len(gr_lst)):
        plt.subplot(6,5,c+1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.imshow(gr_lst[c])

    
    



