from skimage import io
import cv2
import glob
import os
g = glob.glob('color_dataset/*.*')
os.mkdir('bin/first/second')
img_=1
for i in g:
    a = cv2.imread(i)
    re = cv2.resize(a, (400,400))
    img_ += 1
    cv2.imwrite('bin/first/second' +str(img_)+ '.jpg', re)
    #cv2.imwrite('gray_dataset/' +str(img_num)+ '.jpg', gray)
    cv2.imshow('image', re)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
print('done')
