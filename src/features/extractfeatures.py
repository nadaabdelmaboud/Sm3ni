from features.Blobs import *
from features.flagextraction import *
from features.Beams import *
from features.Blobs_VerticalLines import *
import numpy as np
from skimage.morphology import thin


def checkLines(width, height, linesCount, maxSpace, BblobsCount, WblobsCount):
    if(height > 2.5*maxSpace and linesCount >= 1 and (BblobsCount == 1 or WblobsCount == 1)):
        return True
    return False


def extractFeatures(symbol, maxSpace):
    features = []

    removed, count, blobOut, lines, maxProjectedLine = detectBeams(
        symbol, maxSpace)
    nOfBlack, BlackCentroids, Bblobs = detectBlackBlob(blobOut, maxSpace)
    nOfWhite = 0
    WhiteCentroids = []
    Wblobs = []
    if(nOfBlack == 0):
        nOfWhite, WhiteCentroids, Wblobs = detectWhiteBlob(blobOut, maxSpace)
    nOfChords, ChordsCentroids, Cblobs = detectOneLinedChords(
        blobOut, maxSpace)
    nOfBlack, BlackCentroids, Bblobs, nOfWhite, WhiteCentroids, Wblobs = setBlobsProperties(
        nOfBlack, BlackCentroids, nOfWhite, WhiteCentroids, nOfChords, ChordsCentroids, Bblobs, Wblobs, Cblobs)
    upOrdown = setBlobsWithLines(nOfBlack, BlackCentroids, nOfWhite,
                                 WhiteCentroids, lines, maxProjectedLine, symbol.shape[0], maxSpace)

    s = symbol.copy()

    for B in Bblobs:
        minr, minc, maxr, maxc = B
        s = 1 - symbol
        s[minr:maxr, minc:maxc] = 0

    for W in Wblobs:
        minr, minc, maxr, maxc = W
        s = 1 - symbol
        s[minr:maxr, minc:maxc] = 0

    if (len(Bblobs) == 0 and len(Wblobs) == 0):
        s = 1 - symbol

    countV = 0
    countInvertedV = 0
    if(checkLines(symbol.shape[1], symbol.shape[0], len(lines), maxSpace, len(Bblobs), len(Wblobs))):
        skelImg = thin(s)
        countV = findV(skelImg)
        countInvertedV = findVinverted(skelImg)
    if(upOrdown == 1):
        countInvertedV = 0
    elif(upOrdown == 0):
        countV = 0
    else:
        countV = 0
        countInvertedV = 0
    features.append(nOfBlack)
    features.append(nOfWhite)
    features.append(count)
    features.append(upOrdown)
    features.append(countV)
    features.append(countInvertedV)
    features.append(symbol.shape[0]/symbol.shape[1])
    return features, BlackCentroids, WhiteCentroids
