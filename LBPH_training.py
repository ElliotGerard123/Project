
import cv2
import numpy as np
from matplotlib import pyplot as plt


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
    img = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)
    height,width= img.shape
    img_lbp = np.zeros((height, width),
                       np.uint8)
    # for each position of height-width
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = matrix_LBP(img, i, j)

    cv2.imshow('this is end picture', img_lbp)
    # split image into multiple grids x,y
    y1 = 0
    H = height // 8
    W = width // 8
    x = np.array(64, dtype=object)

    i = 0
    allCounts = []
    allBinLocs = []
    for y in range(0, height, H):
        for x in range(0, width, W):
            y1 = y + H
            x1 = x + W
            tiles = img_lbp[y:y + H, x:x + W]
            allCounts.append(tiles)
            pic = cv2.rectangle(img_lbp, (x, y), (x1, y1), (10, 30, 20))
            cv2.imshow('', pic)

    h = cv2.calcHist(allCounts, [0], None, [256], [0, 256])
    plt.plot(h)
    plt.show()
    return h

"""We just train an specific image, in a future will be a set of images called testing_set"""



###########################HERE WE ARE CALLING FUNCTIONS##############################
########In this case will be just one image###########################################
if __name__ == '__main__':
    path = 'images/Gerard/Gerard1.jpeg'
    image = cv2.imread(path)



    #convert to gray

    ## here we do all the process
    Histograms(image)







    print("LBP Program is finished")

    #extracting histogram



cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)