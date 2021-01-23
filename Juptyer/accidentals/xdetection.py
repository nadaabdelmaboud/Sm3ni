
import cv2
import numpy as np
from commonfunctions import *
from skimage.morphology import skeletonize


def line_intersection(line1, line2,x,y):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False ,x,y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return True,x,y

'''
@gray : gray image with the symbol white and the background black
@isX : if the symbol is x the function returns true else it returns false
'''
def isSymbolX(gray):
    threshold = 100
    maxLineGap = 1
    minLineLength = 1
    thresholdedIMG = np.zeros(gray.shape)
    skelImg = np.zeros(gray.shape,dtype=np.uint8)
    rows,cols = gray.shape
    
    
    #thresholding image
    thresholdedIMG[gray>threshold] = 1
    
    #skeltonizing image
    thresholdedIMG = skeletonize(thresholdedIMG)
    skelImg[thresholdedIMG ==True] = 255
    skelImg[thresholdedIMG == False] = 0
    
    #detecting lines
    minLineLength=1
    minvotes = skelImg.shape[0]//4
    lines = cv2.HoughLinesP(skelImg,1,np.pi/180,minvotes,minLineLength=minLineLength,maxLineGap=maxLineGap)

    
    empty = False
    isX = False
    x=0
    y=0
    diagonalLines = []
        
    #handling if no lines are detected
    try:
        if not len(lines) :
            lines = np.array([])
    except Exception as e:
            empty = True
            
    #filtering lines and elemenating vertical and horizontal lines
    if(not(empty) and lines.shape):
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                if(x2-x1!=0 and y2-y1!=0):
                    diagonalLines.append([x1,y1,x2,y2])
   
    #finding intersection between any two lines perpendecular on each other
    if(len(diagonalLines)>=2):
        for i in range (0,len(diagonalLines)):
            for j in range (0,len(diagonalLines)):
                if i!=j:
                    x1,y1,x2,y2 = diagonalLines[i]
                    m1 = ((y2-y1)/(x2-x1))
                    line1=np.array([[x1,y1],[x2,y2]],dtype=np.int)
                    x1,y1,x2,y2 = diagonalLines[j]
                    m2 = ((y2-y1)/(x2-x1))
                    line2=np.array([[x1,y1],[x2,y2]],dtype=np.int)
                    if(m1!=m2 and m1*m2 <0 ):
                        isX , x , y = line_intersection(line1,line2,x,y)
    return isX



#reading image
img = cv2.imread('images/img/17.png')
#converting to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#inverting image
gray = 255-gray
#check if it's x
print(isSymbolX(gray))