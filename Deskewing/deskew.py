import numpy as np
import cv2
from scipy.ndimage import interpolation as inter

def deskew(img):
    initAngle = 45
    angles = np.arange(-initAngle, initAngle + 1, 1)
    scoreArray = np.zeros(92)
    for i in angles:
        data = inter.rotate(img, i, reshape=False, order=0)
        sumRow = np.sum(data, axis=1)
        score = np.sum((sumRow[1:] - sumRow[:-1]) ** 2)
        scoreArray[i+45] = score

    data = inter.rotate(img, 90, reshape=True, order=0)
    sumRow90 = np.sum(data, axis=1)
    score90 = np.sum((sumRow90[1:] - sumRow90[:-1]) ** 2)
    scoreArray[91] = score90

    scoreArray = np.array(scoreArray)
    trueAngle = np.where(scoreArray == max(scoreArray))[0][0] - 45
    if(trueAngle == 46):
        rotated = inter.rotate(img, 90, reshape=True, order=0)
    else:
        rotated = inter.rotate(img, trueAngle, reshape=False, order=0)
        print(trueAngle)
    return rotated