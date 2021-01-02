import cv2
import numpy as np
from commonfunctions import *
from skimage.morphology import skeletonize


'''
@gray: the gray scale image with the symbol white and the background black
@flag: true if there is vertical lines , flase if there is no verticl lines
@X: list has all the x coordinates (columns) of the vertical lines
@vericalLines: list containing the start point and end point of the detected vertical lines
'''
def getVirtivalLines(gray):
    threshold = 100
    maxLineGap = 1
    minLineLength = 1
    
    print("thresholding image ...")
    thresholdedIMG = np.zeros(gray.shape)
    skelImg = np.zeros(gray.shape,dtype=np.uint8)
    
    
    
    print("skeltonizing image")
    thresholdedIMG[gray>threshold] = 1
    thresholdedIMG = skeletonize(thresholdedIMG)
    skelImg[thresholdedIMG == True] = 255
    skelImg[thresholdedIMG == False] = 0
    
    
    lines = cv2.HoughLinesP(skelImg,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    
    X=[]
    verticalLines = []
    empty = False
    
    try:
        if not len(lines) :
            lines = np.array([])
    except Exception as e:
            empty = True
            return False,X,verticalLines
    
    if(not(empty) and lines.shape):
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                flag = X.index(x1) if x1 in X else -1
                if(x1==x2 and flag==-1):
                    print(lines[x])
                    X.append(x1)
                    verticalLines.append([x1,y1,x2,y2])
                    #cv2.line(binary,(x1,y1),(x2,y2),255,2)
                if(x1 == x2 and flag!=-1):
                    nx1,ny1,nx2,ny2 = verticalLines[flag]
                    newy1 = max(y1,ny1)
                    newy2 = min(y2,ny2)
                    verticalLines[flag]=[x1,newy1,x2,newy2]
                    cv2.line(binary,(x1,newy1),(x2,newy2),255,2)
                    print(verticalLines[flag])
    
    return True,X,verticalLines




print("reading image ...")
img = cv2.imread('images/img/14.png')

print("converting to gray scale ...")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = 255-gray


rows,cols = gray.shape
binary = np.zeros((rows,cols)).astype('uint8')
    
flag,Xcoordinates,verticalLines = getVirtivalLines(gray)
if(flag == True):
    for x in range(0,len(verticalLines)):
        x1,y1,x2,y2 = verticalLines[x]
        cv2.line(binary,(x1,y1),(x2,y2),255,2)

            

show_images([img,gray,binary],["original image","resized","verticallines"])
