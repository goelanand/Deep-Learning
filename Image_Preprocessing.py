import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import tifffile as tif
import skimage
import cv2
from skimage.filters import gaussian
from skimage import img_as_ubyte



print('start')
imagelist=[]
path = glob.glob('E:/Pending2uploadingihub/Dataset/Apparel_Random_Dataset/*.*')
for file in path:
    a = cv2.imread(file)
    imagelist.append(a)
    plt.imshow(imagelist[file])
    