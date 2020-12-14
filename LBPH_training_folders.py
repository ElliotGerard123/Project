
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from PIL import Image
import time


def binary_pixel(img, center, x, y):
    new_value = 0

    try:
        #WE ARE APPLYING THRESHOLD
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. valuess
        # present at boundaries.
        pass

    return new_value

# Function for calculating LBP
def matrix_LBP(img, x, y):
    center = img[x][y]
    #x = height y= width
    matrixPixels = []
    # we set each value of the matrix ( 0 or 1 )
    # top_left
    matrixPixels.append(binary_pixel(img, center, x - 1, y - 1))

    # top
    matrixPixels.append(binary_pixel(img, center, x - 1, y))

    # top_right
    matrixPixels.append(binary_pixel(img, center, x - 1, y + 1))

    # right
    matrixPixels.append(binary_pixel(img, center, x, y + 1))

    # bottom_right
    matrixPixels.append(binary_pixel(img, center, x + 1, y + 1))

    # bottom
    matrixPixels.append(binary_pixel(img, center, x + 1, y))

    # bottom_left
    matrixPixels.append(binary_pixel(img, center, x + 1, y - 1))

    # left
    matrixPixels.append(binary_pixel(img, center, x, y - 1))

    #CONVERTING BINARY to DECIMAL

    matrixPix = ''.join(map(str, matrixPixels))

    new_value = int(matrixPix ,2)
    #print("This is bin",matrixPix,"This is decimal",new_value)

    #converting central ( decimal )
    for i in range(len(matrixPixels)):

        if i == 4:
            matrixPixels[i] = new_value


    #print(matrixPixels)
    return new_value
"""There is another method check website:
    
    https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/
    --------
    """


def Histograms(img):
    # Create a numpy array as
    # the same height and width
    # of RGB image
    height, width= img.shape
    img_lbp = np.zeros((height, width),
                       np.uint8)
    # for each position of height-width
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = matrix_LBP(img, i, j)

    #cv2.imshow('this is end picture', img_lbp)
    # split image into multiple grids x,y
    y1 = 0
    H = height // 8
    W = width // 8
    x = np.array(64, dtype=object)

    i = 0
    allCounts = []
    allhistograms = []
    for y in range(0, height, H):
        for x in range(0, width, W):
            y1 = y + H
            x1 = x + W
            tiles = img_lbp[y:y + H, x:x + W]
            allCounts.append(tiles)

            pic = cv2.rectangle(img_lbp, (x, y), (x1, y1), (10, 30, 20))
            #x = cv2.imshow('', pic)

    h = cv2.calcHist(allCounts, [0], None, [256], [0, 256])
    allhistograms.append((h))
    print(allhistograms)
    #plt.plot(h)
    #plt.show()

    cv2.waitKey(0)
    return allhistograms



def read_images():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images/Gerard")

 #   File = 'images/Gerard/*.*'

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                try:
                    path = os.path.join(root, file)
                    print(file)
                    pil_image = Image.open(path)
                    #pil_image.show()
                    image = cv2.imread(file)


                    open_cv_image = np.array(pil_image)
                    # Convert RGB to BGR
                    cvimage = open_cv_image[:, :, ::-1].copy()
                    #cv2.imshow('', cvimage)
                    # convert to gray
                    img_gray = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY)
                    ## here we do all the process

                    Histograms(img_gray)

                except Exception as e:
                    print(str(e))


"""We will have to go over all the images not just one. 
To do that we will create another function to and label the pictures.
Then call the rest of the functions to have the histogram of the specific function"""



###########################HERE WE ARE CALLING FUNCTIONS##############################
########In this case will be just one image###########################################
if __name__ == '__main__':
    tic = time.perf_counter()

    read_images()


    print("LBP Program is finished")
    toc = time.perf_counter()

    print(f"final time {toc - tic:0.4f} seconds")

    #extracting histogram



cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)