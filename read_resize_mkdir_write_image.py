#Using os library to walk through folders
import os
import cv2
import glob

#Download the image from github repo and replace with your local path
# read the image and create new folder and enlarge(Resize) and write

os.mkdir('D:/Spider_new_projects/Image_Segmentation/Image_Preprocessing/bin/a')

path = glob.glob('D:/Spider_new_projects/Image_Segmentation/Image_Preprocessing/color_dataset/*.*')

img = 1
for x in path:
    c = cv2.imread(x)
    r = cv2.resize(c, (500,500))
    cv2.imwrite('D:/Spider_new_projects/Image_Segmentation/Image_Preprocessing/bin/a/' +str(img)+ '.jpg', r)
    img += 1
    print('writing-->' , img,'...Done.')
    cv2.imshow('image',r)
    cv2.waitKey(200)
    cv2.destroyAllWindows()
print('Successfully Completed', img-1, 'Files....')