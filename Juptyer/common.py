import cv2
import skimage.io as io
import numpy as np
import scipy
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from scipy.signal import convolve2d
from statistics import mode,variance
from math import sqrt
from skimage.measure import find_contours
import matplotlib.pyplot as plt 
from skimage import data, color, img_as_ubyte 
from skimage.feature import canny 
from skimage.transform import hough_ellipse 
from skimage.draw import ellipse_perimeter 
from skimage.draw import rectangle
from skimage.morphology import disk
from scipy.spatial.distance import euclidean
from skimage.util.shape import view_as_windows
from pip._internal import main as install
from pylab import imshow, gray, show 
from math import pi
from scipy.ndimage import interpolation as inter

def showBinaryImg(img):
    cv2.imshow("Images",img*255)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
def showImg(img):
    cv2.imshow("Images",img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
def opening(img,ST):
    imgEroded=cv2.erode(img,ST)
    return cv2.dilate(imgEroded,ST)
def closing(img,ST):
    imgDilated=cv2.dilate(img,ST)
    return cv2.erode(imgDilated,ST)