import cv2
import glob
import matplotlib.pyplot as plt

path = glob.glob('color_dataset/*.*')
for x in path:
    co=cv2.imread(x)
    cv2.imshow('ing', co)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
        
#Converting Color images to Gray      
cg=[]
gr_f=1
for a in path:
    dc = cv2.imread(a)
    gr=cv2.cvtColor(dc, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_dataset/' +str(gr_f)+ '.jpg', gr)
    #cv2.imwrite('gray_dataset/' +str(img_num)+ '.jpg', gray)
    gr_f +=1

#Resizing     
gray_path = glob.glob('gray_dataset/*.*')
gr_l=[]
fileno = 1
for i in gray_path:
    g = cv2.imread(i)
    image = cv2.resize(g, (170, 170))
    cv2.imwrite('resized_dataset/' +str(fileno)+ '.jpg' , image)
    fileno += 1

#checking new resize images
rel=[]
for p in (glob.glob('resized_dataset/*.*')):
    rep = cv2.imread(p)
    rel.append(rep)
    for q in range(len(rel)):
        plt.subplot(5,5,q+1)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.imshow(rel[q])
        
        
    
