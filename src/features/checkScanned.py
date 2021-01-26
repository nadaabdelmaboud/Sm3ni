
import cv2
import numpy as np
from skimage.morphology import skeletonize

'''
@binarized:  binarized image with the symbol white and the background black
@flag: true if there is vertical lines , flase if there is no verticl lines
'''
def getHorizontalLines(binarized):
    
    h,w=binarized.shape
    maxLineGap = 1
    minLineLength = w*0.4
    
    neg = 1 - binarized
    thresholdedIMG = skeletonize(neg)
    skelImg = np.zeros(binarized.shape)
    skelImg[thresholdedIMG == True] = 255
    skelImg[thresholdedIMG == False] = 0
    skelImg = np.uint8(skelImg)
    lines = cv2.HoughLinesP(skelImg,1,np.pi/180,10,minLineLength=minLineLength,maxLineGap=maxLineGap)
    
    
    try:
        if not len(lines) :
            lines = np.array([])
    except Exception as e:
            empty = True
            return False
    
    return True



            
