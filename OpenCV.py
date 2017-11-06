import numpy as np
import cv2
import matplotlib
import sys
import matplotlib.pyplot as plt

def scale(imgs, factor):
    result = imgs.dot(factor)
    return result

# Load an color image in grayscale
img = cv2.imread('Macclaren.png', 1)  # load image as colour image Arg 0 = greyscale, 1 = RGB

#Understand image properties
# print img.item(10,10,2) # print the value of pixel 100,100 RED Channel to debug console
# print img.shape # show the dimensions and number of channels
# roi=img[50:500,100:500] # take a region of interest in [rows, columns]

img2 = cv2.imread('JLR L2P 2016.jpg', 1)  # Load a second image
img2 = cv2.resize(img2, (1920, 1086))  # Resize the image ready to add two images

if img.shape == img2.shape:  # Chck image sizes are equal, ensure if and else are same indent level
    print "Image sizes equal"
else:
    print "Images not equal"
    sys.exit()
# ToDo write code to add images, subtract images and alpha blend
imgProd = img + img2  # add images together
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray scale
gray_image = scale(gray_image, 1.5)

# ToDo write code to graph intensity of colour channels
#colour = ('b', 'g', 'r')
#for i, colour in enumerate(colour):
#    hist = cv2.calcHist([img], [i], None, [256],[0, 256])  # using open CV function to calculate histogram
#  40x faster than numpy
#plt.plot(hist, color=colour)
#plt.xlim([0, 256])



# ToDo write code to add gaussian noise]
row, col= gray_image.shape
sigma = 150
noise = sigma * np.random.randn (row,col) # Generate a random data set of normal distribution RANDN
hist, bins = np.histogram(noise,bins = 100) #  Create the histogram and return the bins and histogram
centre = (bins[:1]+ bins[1:])/2 # deifne the center of the histogram
plt.bar(centre, hist, align = 'center',width=np.diff(bins)) # plot

gray2_image = gray_image + noise

# show and manage windows
# plt.show()
# plt.close()
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', gray2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
