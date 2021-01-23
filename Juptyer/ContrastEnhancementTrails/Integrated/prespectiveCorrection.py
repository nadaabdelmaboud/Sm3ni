


import numpy as np
import cv2


def prespectiveCorrection(RotatedImage):
    #rotated image is the image returned from the deskew directly
    RotatedImage = RotatedImage*255
    rows,cols = RotatedImage.shape
    edges = cv2.Canny(RotatedImage,50,150,apertureSize = 3)
    
    maxLineGap = 10
    minLineLength = 60
    prespectiveSlopeThreshold = 0.05
    flag=False
    
    
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    binary = np.zeros((rows,cols)).astype('uint8')
    
    s = 0
    countLines = 0
    countOfVerticalAndhorizontalLines = 0
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(binary,(x1,y1),(x2,y2),255,2)
            m = ((y2-y1)/(x2-x1))
            if(abs(m)==np.inf or m==0):
                countOfVerticalAndhorizontalLines+=1
            if(abs(m) != np.inf):
                countLines+=1
                s+=m
                
    avrgSlope = s/countLines 
    percentage = (countOfVerticalAndhorizontalLines/len(lines))*100
              

    if(abs(avrgSlope)>prespectiveSlopeThreshold and percentage<80):
        
        SE = np.ones([50,50])      
        closedImage = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, SE)
        
    
        contours, hierarchy = cv2.findContours(closedImage,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
        
        for box in cnts:
            epsilon = 0.02*cv2.arcLength(box,True)
            approx = cv2.approxPolyDP(box,epsilon,True) 
            for i in range(0,len(approx)):
                RotatedImage = cv2.circle(RotatedImage, (approx[i,0,0],approx[i,0,1]), 2, 200, 2)
            if(len(approx)==4):
                flag=True
                points=approx
                break
            
        if flag==True:
            
            margin = 20
            
            points[0,0,0]+=margin
            points[0,0,1]-=margin
            points[1]-=margin
            points[2,0,0]-=margin
            points[2,0,1]+=margin
            points[3]+=margin
            
            
            cv2.drawContours(RotatedImage, [points], -1, (0, 255, 0), 2)
            
            rect = np.zeros((4, 2), dtype = "float32")
            s1 = points.sum(axis = 1)
            s = s1.sum(axis=1)
            
            
            rect[0] = s1[np.argmin(s)]
            rect[2] = s1[np.argmax(s)]
            diff = np.diff(s1, axis = 1)
            rect[1] = s1[np.argmin(diff)]
            rect[3] = s1[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
            		[0, 0],
            		[maxWidth - 1, 0],
            		[maxWidth - 1, maxHeight - 1],
            		[0, maxHeight - 1]], dtype = "float32")
                
        
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(RotatedImage, M, (maxWidth, maxHeight)) 
            
        else:warped = RotatedImage
            
    else:
        warped = RotatedImage
        
    return warped
