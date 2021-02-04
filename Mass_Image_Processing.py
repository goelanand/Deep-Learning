import cv2
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte
import matplotlib.pyplot as plt


#select the path
# We have stored some random apparel color images and 
# trying to convert all the images in gray scale images


# First download the images from my github 
# replace the url below with your local repo path

#file = 'https://github.com/azhar2ds/DataSets/tree/master/Apparel_Random_Dataset/*.*'



p = glob.glob('color_dataset/*.*')
ol=[]
img_num=1
for x in p:
    img = cv2.imread(x)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    makeborder = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REPLICATE)
    medianblur = cv2.medianBlur(img, 5)
    guassian_blur = cv2.GaussianBlur(img, (15,15), 0)
    bilateral = cv2.bilateralFilter(img, 9, 475, 475)
    biColor = cv2.bilateralFilter(img, 20, 90,8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    smoothed_dataset = img_as_ubyte(gaussian(img, sigma=5, mode='constant', cval=0.0))
    resized_dataset = cv2.resize(img, (170, 170))
    cv2.imwrite('gray_dataset/' +str(img_num)+ '.jpg', gray)
    cv2.imwrite('makeborder/' +str(img_num)+ '.jpg', makeborder)    
    cv2.imwrite('medianblur/' +str(img_num)+ '.jpg', medianblur)
    cv2.imwrite('guassian_blur/' +str(img_num)+ '.jpg', guassian_blur)
    cv2.imwrite('bilateral/' +str(img_num)+ '.jpg', bilateral)
    cv2.imwrite('biColor/' +str(img_num)+ '.jpg', biColor)
    cv2.imwrite('hsv/' +str(img_num)+ '.jpg', hsv)
    cv2.imwrite('sobelx/' +str(img_num)+ '.jpg', sobelx)
    cv2.imwrite('smoothed_dataset/' +str(img_num)+ '.jpg', smoothed_dataset)
    cv2.imwrite('resized_dataset/' +str(img_num)+ '.jpg', resized_dataset)
    img_num += 1


print('done')
