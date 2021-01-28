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
        # print(trueAngle)
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

def isReversed(segContours,maxSpace,maxLenStaffLines):
    found = False
    print(segContours)
    print("maxLenStaffLines",maxLenStaffLines)
    print("maxSpace",maxSpace)
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

    print("soulKeyDist",soulKeyDist)
    if (soulKeyDist > 0.5 * maxLenStaffLines):
        return True
    else:
        return False
    