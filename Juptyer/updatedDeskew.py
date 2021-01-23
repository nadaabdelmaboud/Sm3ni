import numpy as np
from scipy.ndimage import interpolation as inter

def deskew(img):
    scoreArray = np.zeros(362)
    i = 0
    while i <= 360:
        data = inter.rotate(img, i, reshape=False, order=0)
        sumRow = np.sum(data, axis=1)
        score = np.sum((sumRow[1:] - sumRow[:-1]) ** 2)
        scoreArray[i] = score
        print(i)
        i+=1
    scoreArray = np.array(scoreArray)
    trueAngle = np.where(scoreArray == max(scoreArray))[0][0]
    rotated = inter.rotate(img, trueAngle, reshape=True, order=0)
    return rotated

