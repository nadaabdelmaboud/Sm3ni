import skimage.measure as measure
from skimage.morphology import disk
from features.Blobs_VerticalLines import *

#CREDITS TO Learn python and open cv documentation

def fillHole(img):
    im_in = img
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out


def opening(img, ST):
    imgEroded = cv2.erode(img, ST)
    return cv2.dilate(imgEroded, ST)


def closing(img, ST):
    imgDilated = cv2.dilate(img, ST)
    return cv2.erode(imgDilated, ST)


def testBlob(blob, maxSpace, widthError):
    width = blob.bbox[3]-blob.bbox[1]+widthError
    height = blob.bbox[2]-blob.bbox[0]
    minEccentricity = .3
    maxEccentricity = .9
    minArea = .5*(maxSpace**2)
    maxArea = 4*(maxSpace**2)
    minMajorAxis = .9*maxSpace
    maxMajorAxis = 3.2*maxSpace
    minMinorAxis = .5*maxSpace
    maxMinorAxis = 2*maxSpace
    minSolidity = .8
    if(blob.solidity >= minSolidity and blob.area > minArea and blob.area < maxArea and blob.eccentricity > minEccentricity and blob.eccentricity < maxEccentricity and width > minMajorAxis and width < maxMajorAxis and height > minMinorAxis and height < maxMinorAxis):
        return True
    return False


def detectBlob(symbol, maxSpace, widthError):
    # detect number of blobs in the contour - detect centroid of blobs
    labels = measure.label(symbol)
    if(labels.ndim < 2):
        return 0, np.array([]), []
    props = measure.regionprops(labels)
    isBlob = False
    numBlobs = 0
    centriods = []
    blobs = []
    for prop in props:
        blobSymbol = []
        isBlob = testBlob(prop, maxSpace, widthError)
        if(isBlob):
            minr, minc, maxr, maxc = prop.bbox
            # blobSymbol=symbol[minr:maxr,minc:maxc]
            blobs.append(prop.bbox)
            centriods.append(prop.centroid)
            numBlobs = numBlobs+1
    centriods = np.array(centriods)
    return numBlobs, centriods, blobs


def detectBlackBlob(contour, maxSpace):
    width = []
    if(contour.shape[0] > 3*maxSpace or contour.shape[1] < 1.2*contour.shape[0]):
        contour, peaksMids, maxProjection, width = removeSymbolVerticalLines(
            contour, maxSpace)
    contour = np.array(contour)
    contour = 1-contour
    widthError = 0
    if(len(width) > 0):
        widthError = width[0]
    return detectBlob(contour, maxSpace, widthError)


def detectWhiteBlob(contour, maxSpace):
    contour = np.array(contour)
    img3 = contour
    imgTmp = np.ones((img3.shape[0]+20, img3.shape[1]+20))
    imgTmp[10:imgTmp.shape[0]-10, 10:imgTmp.shape[1]-10] = img3
    img3 = 1-imgTmp
    img3 = (img3*255).astype("uint8")
    ST = np.ones((1, 10))
    img3 = closing(img3, ST)
    img3 = img3 > 100
    img3 = img3*1
    img3 = 1-img3
    img3 = (img3*255).astype("uint8")
    img3 = fillHole(img3)
    img3 = img3 > 100
    img3 = img3*1
    img3 = 1-img3
    if(contour.shape[0] > 3*maxSpace or contour.shape[1] < 1.2*contour.shape[0]):
        contour, peaksMids, maxProjection, width = removeSymbolVerticalLines(
            img3, maxSpace)
    else:
        contour = img3
    contour = np.array(contour)
    contour = 1-contour
    num, centroids, blobs = detectBlob(contour, maxSpace, 0)
    if(num > 0):
        centroids[0][0] = centroids[0][0]-10
        centroids[0][1] = centroids[0][1]-10
    return num, centroids, blobs


def detectOneLinedChords(contour, maxSpace):
    contour = np.array(contour)
    if(contour.shape[0] > 3*maxSpace or contour.shape[1] < 1.2*contour.shape[0]):
        contour = removeSymbolVerticalLines(contour, maxSpace)[0]
    contour = np.array(contour)
    contour = 1-contour
    ST = np.ones((1, int(maxSpace)))
    opened = opening((contour*255).astype("uint8"), ST)
    opened = opened > 100
    opened = opened*1
    return detectBlob(opened, maxSpace, 0)


def isDublicate(blob1, blob2):
    error = 3
    maxX = 0
    minX = 0
    if(blob2[0] > blob1[0]):
        maxX = blob2[0]
        minX = blob1[0]
    else:
        maxX = blob1[0]
        minX = blob2[0]
    maxY = 0
    minY = 0
    if(blob2[1] > blob1[1]):
        maxY = blob2[1]
        minY = blob1[1]
    else:
        maxY = blob1[1]
        minY = blob2[1]
    if(minX+error >= maxX and minY+error >= maxY):
        return True
    return False


def removeDublicate(BlackCentroids, Bblobs):
    for i in range(len(BlackCentroids)):
        for j in range(i+1, len(BlackCentroids)):
            if(i < len(BlackCentroids) and j < len(BlackCentroids) and isDublicate(BlackCentroids[i], BlackCentroids[j])):
                BlackCentroids = np.delete(BlackCentroids, j, axis=0)
                del Bblobs[j]
    return BlackCentroids, Bblobs


def removeWhiteDublicate(WhiteCentroids, BlackCentroids, Wblobs):
    for i in range(len(BlackCentroids)):
        for j in range(len(WhiteCentroids)):
            if(i < len(BlackCentroids) and j < len(WhiteCentroids) and isDublicate(BlackCentroids[i], WhiteCentroids[j])):
                WhiteCentroids = np.delete(WhiteCentroids, j, axis=0)
                del Wblobs[j]

    return WhiteCentroids, Wblobs


def setBlobsProperties(nOfBlack, BlackCentroids, nOfWhite, WhiteCentroids, nOfChords, ChordsCentroids, Bblobs, Wblobs, Cblobs):
    if(len(BlackCentroids) != 0):
        if(len(ChordsCentroids) != 0):
            BlackCentroids = np.concatenate([BlackCentroids, ChordsCentroids])
            Bblobs = (Bblobs)+(Cblobs)
            BlackCentroids, Bblobs = removeDublicate(BlackCentroids, Bblobs)
    else:
        BlackCentroids = ChordsCentroids
        Bblobs = Cblobs
    WhiteCentroids, Wblobs = removeWhiteDublicate(
        WhiteCentroids, BlackCentroids, Wblobs)
    return len(BlackCentroids), BlackCentroids, Bblobs, len(WhiteCentroids), WhiteCentroids, Wblobs


def setBlobsWithLines(nOfBlack, BlackCentroids, nOfWhite, WhiteCentroids, lines, maxProjectedLine, height, error):
    # 0 for down , 1 for up , -1 for no blobs
    # error=maxSpace
    thres = height-maxProjectedLine-1
    thres = thres+error
    if(nOfBlack > 0):
        heighestBlob = np.amin(BlackCentroids, 0)[0]
        if(heighestBlob < thres):
            return 1
        else:
            return 0
    elif(nOfWhite > 0):
        heighestBlob = np.amin(WhiteCentroids, 0)[0]
        if(heighestBlob < thres):
            return 1
        else:
            return 0
    else:
        return -1


def detectNonLinearBlob(contour, maxSpace):
    contour = 1-contour
    ST = np.array(disk(int(.5*maxSpace)))
    contour = (contour*255).astype("uint8")
    contour = opening(contour, ST)
    contour = (contour > 100)*1
    return detectBlobNonLinear(contour, maxSpace)


def detectBlobNonLinear(symbol, maxSpace):
    labels = measure.label(symbol)
    if(labels.ndim < 2):
        return 0, np.array([]), np.array([])
    props = measure.regionprops(labels)
    isBlob = False
    numBlobs = 0
    centriods = []
    blobs = []
    for prop in props:
        width = prop.bbox[3]-prop.bbox[1]
        height = prop.bbox[2]-prop.bbox[0]
        if(width > 1.5*maxSpace and height < 2*maxSpace):
            centriods.append(prop.centroid)
            numBlobs = numBlobs+1
    centriods = np.array(centriods)
    return centriods
