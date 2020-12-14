import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from SetDatabase import LBPH_training_folders
from SetDatabase import LBPH_training
"""Just checked how comparing two histogram works. 
Next steps:

    1. How to compare all histograms from trained images with one given image
    2. select multiple images to test them with the given images
"""





def euclideanDistance(array, image):
    x=0
    for i in array:
        compare = cv2.compareHist(image,i,cv2.NORM_L2)

        print(f"this is the the result for each picture{x}", compare)
        x+=1

"""We will have to go over all the images not just one.
To do that we will create another function to and label the pictures.
Then call the rest of the functions to have the histogram of the specific function"""

###########################HERE WE ARE CALLING FUNCTIONS##############################
########In this case will be just one image###########################################

"""in this case these image may be an array of images that we want to reconize, 
We will see in the future. """

arr = []
path = 'images/Gerard/Gerard1.jpeg'
image = cv2.imread(path)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
x=cv2.calcHist(image,[0], None, [256], [0, 256])
#testing_image = LBPH_training.Histograms(image)
#print(testing_image)
path2= 'images/Gerard/Gerard2.jpeg'
image2 = cv2.imread(path2)
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
y =cv2.calcHist(image2,[0], None, [256], [0, 256])

arr.append(y)

path3= 'images/Jordi_Papa/Jordi_Papa1.jpg'
image3 = cv2.imread(path3)
cv2.imshow('', image3)
image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
z =cv2.calcHist(image3,[0], None, [256], [0, 256])
arr.append(z)
arr.append('gerard')
print(arr)
plt.show()
#p = LBPH_training_folders.read_images()
#print(p)
euclideanDistance(arr,x)
print("LBP Program is finished")

# extracting histogram


cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)