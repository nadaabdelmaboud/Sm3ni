import numpy as np
import cv2 as cv2


def verticalProj(img):
    img = np.array(img)
    imgTmp = np.ones((img.shape[0], img.shape[1]+2))
    imgTmp[:, 1:imgTmp.shape[1]-1] = img
    img = imgTmp
    proj = np.sum(img, 0)
    maxProjection = img.shape[0]-np.min(proj)
    result = np.zeros(img.shape)
    for col in range(img.shape[1]):
        result[0:int(proj[col]), col] = 1
    return result, maxProjection


def calcVerticalLinesPos(colHist, maxProjection):
    colHist = np.array(colHist)
    m = colHist.shape[1]
    thres = int(maxProjection*0.6)
    thres = colHist.shape[0]-thres-1
    start = end = 0
    width = []
    peaksMid = []
    for i in range(m-1):
        if(colHist[thres][i] != colHist[thres][i+1]):
            if colHist[thres][i] == 1:
                start = i
            else:
                end = i+1
                peaksMid.append(int((start+end)/2))
                width.append((end-start))

    return np.array(width), np.array(peaksMid)


def removeVerticalLines(imgOriginal, midPoint, curWidth):
    imgOriginal = np.array(imgOriginal)
    thresPixel = curWidth
    tmpWidth = curWidth
    curWidth = int(curWidth/2)+1
    for i in range(imgOriginal.shape[0]):
        pixelSum = imgOriginal[i:i+1, midPoint -
                               curWidth:midPoint+curWidth].sum()
        pixelSum = (tmpWidth)-pixelSum
        if(pixelSum <= thresPixel):
            imgOriginal[i:i+1, midPoint-curWidth:midPoint+curWidth] = 1
    return imgOriginal


def applyRemoving(width, peaksMids, img):
    img = np.array(img)
    for i in range(len(peaksMids)):
        img = removeVerticalLines(img, peaksMids[i], width[i])
    return img


def removeSymbolVerticalLines(contour, maxSpace):
    contour = np.array(contour)
    Binarized = contour
    result, maxProjection = verticalProj(Binarized)
    if(maxProjection < 2*maxSpace):
        return contour, np.array([]), 0, np.array([])
    width, peaksMids = calcVerticalLinesPos(result, maxProjection)
    removedImg = applyRemoving(width, peaksMids, Binarized)
    return removedImg, peaksMids, maxProjection, width
