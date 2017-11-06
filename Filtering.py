import numpy as np
import cv2
import matplotlib
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import scipy


def scale(imgs, factor):
    result = imgs.dot(factor)
    return result


# define the kernel for gaussian filter
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Load an color image in grayscale
img = cv2.imread('Macclaren.png', 0)  # load image as colour image Arg 0 = greyscale, 1 = RGB

# Create a gaussian filter
hsize = 81.0  # define the size of the gaussian filter
bsize = 81
sigma = 5  # deifne the sigma, i.e how wide is the width
img = cv2.GaussianBlur(img, (bsize, bsize), sigma)

filter = matlab_style_gauss2D((hsize, hsize), sigma)  # create filter

fig = plt.figure()  # create a plotting  instance
ax = fig.gca(projection='3d')  # plot in 3D for vieiwng

# Make Data
X = np.arange(-10, 10, float(20 / hsize))  # Create the X axis data, must be same length as gaussian size
Y = np.arange(-10, 10, float(20 / hsize))  # Create the Y axis data, must be same length as gaussian size

X, Y = np.meshgrid(X, Y)  # port X, Y to mesh grid...

# Create the plotting surface
surf = ax.plot_surface(X, Y, filter, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# set plot limits and format
ax.set_zlim(np.amin(filter), np.amax(filter))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)  # size colour bar for resolution

# scipy.ndimage.convolve(img, h, mode='nearest')
plt.show()

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
