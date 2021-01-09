import cv2
import numpy as np
from statistics import mode,variance
from skimage.measure import find_contours

def rowProjection(img):
    proj = np.sum(img,1)
    maxProjection = np.max(proj)
    result = np.zeros(img.shape)
    # Draw a line for each row
    for row in range(img.shape[0]):
        result[row,0:int(proj[row])]=1
    result=1-result
    cv2.imwrite("result.png",result*255)
    return maxProjection,result

def calcStaffPos(rowHist,maxProjection,thr):
    n,m=rowHist.shape
    thres=int(maxProjection*thr)
    start=end=0
    width=[]
    peaksMid=[]
    for i in range(n-1):
        if(rowHist[i][thres]!=rowHist[i+1][thres]):
            if rowHist[i][thres]==1:
                #end of zeros
                start=i
            else:
                #end of ones
                end=i+1
                peaksMid.append(int((start+end)/2))
                width.append((end-start))
    return np.array(width),np.array(peaksMid)

#estimate staff segment position
def calcSegmentPos(rowHist):
    n,m=rowHist.shape
    thres=20
    start=end=0
    width=[]
    peaksMid=[]
    for i in range(n-1):
        if(rowHist[i][thres]!=rowHist[i+1][thres]):
            if rowHist[i][thres]==1:
                #end of zeros
                start=i
            else:
                #end of ones
                end=i+1
                peaksMid.append(int((start+end)/2))
                width.append((end-start))
    return np.array(width),np.array(peaksMid)

def filterPeaks(peaksMids,widths):
    if(len(peaksMids)<=1):
        return widths,peaksMids
    newPeaks=[]
    newWidth=[]
    sumW=0
    for i in widths:
        sumW+=i
    avgW=sumW/len(widths)
    #if i dont have flase peaks for clusters then this will result in a problem
    var = variance(widths)
    for i in range(len(peaksMids)):
        if(widths[i]>=avgW or (var<1000 and (avgW-widths[i])<20)):
            newPeaks.append(peaksMids[i])
            newWidth.append(widths[i])
    return newWidth,newPeaks

#divide image into segments
def imgStaffSegments(img,segMids,widths,staffS):
    segments=[]
    staffS=int(staffS)
    n,m=img.shape
    print("dim ",n,m)
    for i in range(len(segMids)):
        up=0
        down=n
        if(segMids[i]-int(widths[i]/2)-2*staffS > 0):
            up =segMids[i]- int(widths[i]/2) - 2*staffS
        if(segMids[i]+int(widths[i]/2)+2*staffS<n):
            down=segMids[i]+int(widths[i]/2)+2*staffS
        print(up,down)
        sliced= img[up:down,:]
        if(sliced.shape[0] != 0 and sliced.shape[1] != 0):
            segments.append(sliced)
    return segments

def maxStaffSpace(peaksMids,width):
    maxSpace=0
    for i in range(len(peaksMids)):
        if(i%5<4 and i+1<len(peaksMids)and peaksMids[i+1]-peaksMids[i]>maxSpace):
            maxSpace=peaksMids[i+1]-peaksMids[i]
            maxSpace=maxSpace-(width[i]/2+width[i+1]/2)
    return maxSpace

def segmenting(BinarizedImage):
    #divide image into segments
    #estimate max staff space
    img = 1 - np.copy(BinarizedImage)

    maxProjection,result=rowProjection(img)
    staffWidth,peaksMids=calcStaffPos(result,maxProjection,0.6)
    maxSpace=maxStaffSpace(peaksMids,staffWidth)

    #estimate staff segment position

    segWidth,segMids=calcSegmentPos(result)
    #filter peaks
    segWidth,segMids=filterPeaks(segMids,segWidth)

    #divide image into segments and divide image => return array of images each image has staff segment

    imgSegments = imgStaffSegments(img,segMids,segWidth,maxSpace)
    # print(imgSegments)
    imgindex=0
    for i in imgSegments:
        if(i.shape[0] != 0 and i.shape[1] != 0):
            cv2.imwrite("segments/"+str(imgindex)+".png",i*255)
            imgindex+=1
    return imgSegments

def removeStaffRow(imgOriginal,midPoint,curWidth):
    thresPixel=curWidth
    for i in range(imgOriginal.shape[1]):
        pixelSum= sum(imgOriginal[midPoint-curWidth:midPoint+curWidth,i:i+1])
        if(pixelSum<=thresPixel):
           # print(imgOriginal[midPoint-curWidth:midPoint+curWidth,i:i+1])
            imgOriginal[midPoint-curWidth:midPoint+curWidth,i:i+1]=0
    return imgOriginal

def removeStaffLines(imgSegments):
    #remove staff from array of imgs of segments
    imgsStaffRemoved=[]
    for simg in imgSegments:
        #simg=1-simg
        maxProjection,result=rowProjection(simg)
        staffWidth,peaksMids=calcStaffPos(result,maxProjection,0.6)
        simg=simg.astype(np.float)
        ST=np.ones((2,1))
        simg=cv2.dilate(simg,ST)
        # simg=1-simg
        for i in range(len(peaksMids)):
            simg=removeStaffRow(simg,peaksMids[i],staffWidth[i])
        ST=np.ones((1,2))
        simg=cv2.dilate(simg,ST)
        simg=1-simg
        imgsStaffRemoved.append(simg)

    imgindex=0
    for i in imgsStaffRemoved:
        cv2.imwrite("staffRemoved/"+str(imgindex)+".png",i*255)
        imgindex+=1
    return imgsStaffRemoved
    #return array of images without staff


def checkOverlapped(c,imgContours):
    cXmin,cXmax,cYmin,cYmax = c
    for Xmin,Xmax,Ymin,Ymax in imgContours:
        if(cXmin > Xmin and cXmax < Xmax and cYmin > Ymin and cYmax < Ymax):
            return True
    return False
def filterContours(imgContours):
    filtered=[]
    for c in imgContours:
        if(not checkOverlapped(c,imgContours)):
            filtered.append(c)
    filtered.sort()
    return filtered

def checkPathRightDown(yR,xR,yD,xD,visited,isCorner,skeleton):
    #if valid path update visited and isCorner list
    h,w=visited.shape
    if(visited[yR][xR]):
        return
    #check right path
    nextX=xR
    nextY=yR
    pathR=[]
    pathR.append([nextX,nextY])
    while(nextX+1 != w and nextY+1 != h):
        #check right down
        if(skeleton[nextY+1][nextX+1]):
            nextX=nextX+1
            nextY=nextY+1
            pathR.append([nextX,nextY])
        #check right pixel
        elif(skeleton[nextY][nextX+1]):
            nextX=nextX+1
            pathR.append([nextX,nextY])
        #check down
        elif(skeleton[nextY+1][nextX]):
            nextY=nextY+1
            pathR.append([nextX,nextY])
        #if none exit loop
        else:
            break
    #check if path down is long enough countD>=5
    nextX=xD
    nextY=yD
    countD=0
    while(nextX+1 != w and nextY+1 != h and nextX !=0):
        #check down
        if(skeleton[nextY+1][nextX]):
            nextY=nextY+1
            countD+=1
        #check right down
        elif(skeleton[nextY+1][nextX+1]):
            nextX=nextX+1
            nextY=nextY+1
        #check left down
        elif(skeleton[nextY+1][nextX-1]):
            nextX=nextX-1
            nextY=nextY+1
        #if none exit loop
        else:
            break
    if(len(pathR)>=5 and countD>=4):
        isCorner.append([xD,yD])
        for i in pathR:
            pX=i[0]
            pY=i[1]
            visited[pY][pX]=1

def findVinverted(skeleton):
    h,w=skeleton.shape
    visited = np.zeros((h,w))
    isCorner=[]
    index=0
    pixelIndices = np.where(skeleton==1)
    pixelCount =len(pixelIndices[0])
    for i in range(pixelCount):
        y = pixelIndices[0][i]
        x = pixelIndices[1][i]
        #check if in white pixel is on borders
        if(x+1 == w or y+1 == h or x== 0 or y==0):
            continue
        downR=skeleton[y+1][x+1]
        down=skeleton[y+1][x]
        right=skeleton[y][x+1]
        downL=skeleton[y+1][x-1]
        if (downR and downL):
            checkPathRightDown(y+1,x+1,y+1,x-1,visited,isCorner,skeleton)
        elif (downR and down):
            checkPathRightDown(y+1,x+1,y+1,x,visited,isCorner,skeleton)
        elif (right and downL):
            checkPathRightDown(y,x+1,y+1,x-1,visited,isCorner,skeleton)
        elif (right and down):
            checkPathRightDown(y,x+1,y+1,x,visited,isCorner,skeleton)
    return len(isCorner)


def checkPathRightUp(yR,xR,yU,xU,visited,isCorner,skeleton):
    #if valid path update visited and isCorner list
    h,w=visited.shape
    if(visited[yR][xR]):
        return
    nextX=xR
    nextY=yR
    pathR=[]
    pathR.append([nextX,nextY])
    count=0
    while(nextX+1 != w and nextY != 0):
        #check right pixel
        if(skeleton[nextY][nextX+1]):
            nextX=nextX+1
            pathR.append([nextX,nextY])
        #check right up
        elif(skeleton[nextY-1][nextX+1]):
            nextX=nextX+1
            nextY=nextY-1
            pathR.append([nextX,nextY])
        #check up
        elif(skeleton[nextY-1][nextX]):
            nextY=nextY-1
            pathR.append([nextX,nextY])
        #if none exit loop
        else:
            break
    nextX=xU
    nextY=yU
    countU=0
    while(nextX+1 != w and nextY != 0 and nextX != 0):
        #check up
        if(skeleton[nextY-1][nextX]):
            nextY=nextY-1
            countU+=1
        #check left up pixel
        elif(skeleton[nextY-1][nextX-1]):
            nextX=nextX-1
            nextY=nextY-1
        #check right up
        elif(skeleton[nextY-1][nextX+1]):
            nextX=nextX+1
            nextY=nextY-1
        #if none exit loop
        else:
            break
    if(len(pathR)>=5 and countU>=4):
        isCorner.append([xR,yR])
        for i in pathR:
            pX=i[0]
            pY=i[1]
            visited[pY][pX]=1

def findV(skeleton):
    h,w=skeleton.shape
    visited = np.zeros((h,w))
    isCorner=[]
    index=0
    pixelIndices = np.where(skeleton==1)
    pixelCount =len(pixelIndices[0])
    for i in range(pixelCount):
        y = pixelIndices[0][i]
        x = pixelIndices[1][i]
        #check if in white pixel is on borders
        if(x+1 == w or y+1 == h or x== 0 or y==0):
            continue
        upR=skeleton[y-1][x+1]
        up=skeleton[y-1][x]
        right=skeleton[y][x+1]
        upL=skeleton[y-1][x-1]
        if (upR and upL):
            checkPathRightUp(y-1,x+1,y-1,x-1,visited,isCorner,skeleton)
        elif (upR and up):
            checkPathRightUp(y-1,x+1,y-1,x,visited,isCorner,skeleton)
        elif (right and up):
            checkPathRightUp(y,x+1,y-1,x,visited,isCorner,skeleton)
        elif (right and upL):
            checkPathRightUp(y,x+1,y-1,x-1,visited,isCorner,skeleton)
    return len(isCorner)

def getImageContours(imgsStaffRemoved):
    imgindex=0
    segContours=[]
    for seg in imgsStaffRemoved:
        contours = find_contours(seg, 0.8)
        imgContours=[]
        imgSymbols=[]
        index=0
        for contour in contours:
            x = contour[:,1]
            y = contour[:,0]
            [Xmin, Xmax, Ymin, Ymax] = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]
            imgContours.append([Xmin, Xmax, Ymin, Ymax])

        imgContours = filterContours(imgContours)

        for Xmin,Xmax,Ymin,Ymax in imgContours:
            imgSymbol=seg[int(Ymin):int(Ymax+1),int(Xmin):int(Xmax+1)]
            imgSymbols.append(imgSymbol)
            cv2.imwrite("contours/"+str(imgindex)+"_"+str(index)+".png",imgSymbol*255)
            index+=1

        imgindex+=1
        segContours.append(imgSymbols)

def staffRemoval(BinarizedImage):
    imgsegs = segmenting(BinarizedImage)
    imgsStaffRemoved = removeStaffLines(imgsegs)
    getImageContours(imgsStaffRemoved)