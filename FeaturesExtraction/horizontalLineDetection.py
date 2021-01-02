
import cv2
import numpy as np
from commonfunctions import *
from skimage.morphology import skeletonize


'''
@gray: the gray scale image with the symbol white and the background black
@flag: true if there is vertical lines , flase if there is no verticl lines
@Y: list has all the y coordinates (rows) of the horizontal lines
@horizontalLines: list containing the start point and end point of the detected horizontal lines
'''

def getHorizontalLines(gray):
    threshold = 100
    maxLineGap = 1
    minLineLength = 1
    
    print("thresholding image ...")
    thresholdedIMG = np.zeros(gray.shape)
    skelImg = np.zeros(gray.shape,dtype=np.uint8)
    
    
    
    print("skeltonizing image")
    thresholdedIMG[gray>threshold] = 1
    thresholdedIMG[gray<threshold] = 0
    thresholdedIMG = skeletonize(thresholdedIMG)
    skelImg[thresholdedIMG == True] = 255
    skelImg[thresholdedIMG == False] = 0
    
    
    lines = cv2.HoughLinesP(skelImg,1,np.pi/180,10,minLineLength=minLineLength,maxLineGap=maxLineGap)
    
    Y=[]
    horizontalLines = []
    empty = False
    
    try:
        if not len(lines) :
            lines = np.array([])
    except Exception as e:
            empty = True
            return False,Y,horizontalLines
    
    if(not(empty) and lines.shape):
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                flag = Y.index(y1) if y1 in Y else -1
                if(y1==y2 and flag==-1):
                    Y.append(y1)
                    horizontalLines.append([x1,y1,x2,y2])
                if(y1 == y2 and flag!=-1):
                    nx1,ny1,nx2,ny2 = horizontalLines[flag]
                    newx1 = min(x1,nx1)
                    newx2 = max(x2,nx2)
                    horizontalLines[flag]=[newx1,y1,newx2,y2]
    
    return True,Y,horizontalLines,skelImg




print("reading image ...")
img = cv2.imread('images/beam/h16.png')

print("converting to gray scale ...")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = 255-gray


rows,cols = gray.shape
binary = np.zeros((rows,cols)).astype('uint8')
removed = np.zeros((rows,cols)).astype('uint8')
    
flag,Ycoordinates,horizontalLines,skelImg = getHorizontalLines(gray)
removed = np.copy(skelImg)
if(flag == True):
    for x in range(0,len(horizontalLines)):
        x1,y1,x2,y2 = horizontalLines[x]
        cv2.line(binary,(x1,y1),(x2,y2),255,2)
        cv2.line(removed,(x1,y1),(x2,y2),0,4)

            

show_images([img,gray,skelImg,binary,removed],["original image","gray","skel","horizontalLines","horizontal lines removed"])
