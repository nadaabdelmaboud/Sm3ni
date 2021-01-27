from skimage.measure import label, regionprops
from features.Blobs_VerticalLines import *


def checkBeams(width, height, linesCount, maxSpace):
    if width > 3*maxSpace and height > 2.5*maxSpace and linesCount > 1:
        return True
    return False


def RemoveHorizontalAndDiagonalLines(img, originalImg, maxSpace, linesCount,height,width):
    if(not checkBeams(width, height,linesCount, maxSpace)):
        return img, 0, originalImg

    removed = np.copy(img)
    labelImg = label(img)
    regions = regionprops(labelImg)
    diagonal = []
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        aspectratio = (maxc-minc)/(maxr-minr)
        if(aspectratio >= 1.5 and (maxc-minc) > .4*maxSpace):
            diagonal.append([minr, minc, maxr, maxc])
            removed[minr:maxr, minc:maxc] = 0
            originalImg[minr:maxr, :] = 0
    count = 0
    if len(diagonal) > 0:
        count = 1
        minr, minc, maxr, maxc = diagonal[0]
        col = (maxc+minc)/2
        for i in range(1, len(diagonal)):
            minr, minc, maxr, maxc = diagonal[i]
            if(minc < col and maxc > col):
                count += 1

    return removed, count, originalImg


def detectBeams(img, maxSpace,h,w):
    img = np.array(img)
    orig = np.copy(1-img)
    img, peaksMids, maxProjection, width = removeSymbolVerticalLines(img, maxSpace)

    img = np.array(img)

    removed, count, orig = RemoveHorizontalAndDiagonalLines(1-img, orig, maxSpace, len(peaksMids),h,w)

    return removed, count, 1-orig, peaksMids, maxProjection
