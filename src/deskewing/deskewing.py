from scipy.ndimage import interpolation as inter
import numpy as np


def deskew(img,isSymbol=False,axis=1):
    if isSymbol:
        scoreArray = np.zeros(182)
        i = -45
        maxScore=0
        trueAngle=0
        while i <= 45:
            data = inter.rotate(img, i, reshape=False, order=0)
            sumRow = np.sum(data, axis=axis)
            score = np.sum((sumRow[1:] - sumRow[:-1]) ** 2)
            if(score>maxScore):
                maxScore=score
                trueAngle=i
            i+=1
       
    else:
        scoreArray = np.zeros(362)
        i = 0
        while i <= 360:
            data = inter.rotate(img, i, reshape=False, order=0)
            sumRow = np.sum(data, axis=axis)
            score = np.sum((sumRow[1:] - sumRow[:-1]) ** 2)
            scoreArray[i] = score
            i+=1
        scoreArray = np.array(scoreArray)
        trueAngle = np.where(scoreArray == max(scoreArray))[0][0]
    rotated = inter.rotate(img, trueAngle, reshape=True, order=0)
    return rotated,trueAngle

def rotateBy(img,trueAngle):
    rotated = inter.rotate(img, trueAngle, reshape=True, order=0)
    return rotated


def isReversed(segContours,maxSpace,maxLenStaffLines,heights = []):
    found = False
    if (len(heights) == 0):
        for contour in segContours:
            if(found == True):
                break
            for Xmin,Xmax,Ymin,Ymax in contour:
                soulKeySearching = Ymax - Ymin

                # detecting Soul Key
                if (soulKeySearching > 6*maxSpace):
                    soulKeyDist = Xmin
                    found = True
                    break
    else:
        maxValue = 0
        maxIndex = 0
        indexInSeg = 0
        #getting the clef in the first segment only
        for H in heights[0]:
            if H > maxValue:
                maxValue = H
                maxIndex = indexInSeg
            indexInSeg+=1

        #after detecting clef, now getting the xmin of this contour
        soulKeyDist = segContours[0][maxIndex][0]

    if (soulKeyDist > 0.5 * maxLenStaffLines):
        return True
    else:
        return False
    