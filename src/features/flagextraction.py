import numpy as np
# if color is white => symbol , black => background
# another matrix => visited pixels

# attempt algorithm to find corner
# 1.find suspected point as a begining of corner
# 2.iterate from the right pixel
# 3.if path from this pixel is longer than 4 and dominating direction is DOWN AND RIGHT and does not form a loop(closed path)
#  then this is a corner=> mark as visited
# 4.repeat untill all white pixels are done

# return path length and top two dominating directions


def checkPathRightDown(yR, xR, yD, xD, visited, isCorner, skeleton):
    # if valid path update visited and isCorner list
    h, w = visited.shape
    if(visited[yR][xR]):
        return
    # check right path
    nextX = xR
    nextY = yR
    pathR = []
    pathR.append([nextX, nextY])
    count = 0
    while(nextX+1 != w and nextY+1 != h):
        # check right down
        if(skeleton[nextY+1][nextX+1]):
            nextX = nextX+1
            nextY = nextY+1
            pathR.append([nextX, nextY])
            count += 1
        # check right pixel
        elif(skeleton[nextY][nextX+1]):
            nextX = nextX+1
            pathR.append([nextX, nextY])
        # check down
        elif(skeleton[nextY+1][nextX]):
            nextY = nextY+1
            pathR.append([nextX, nextY])
            count += 1
        # if none exit loop
        else:
            break
    # check if path down is long enough countD>=5
    nextX = xD
    nextY = yD
    countD = 0
    while(nextX+1 != w and nextY+1 != h and nextX != 0):
        # check down
        if(skeleton[nextY+1][nextX]):
            nextY = nextY+1
            countD += 1
        # check right down
        elif(skeleton[nextY+1][nextX+1]):
            nextX = nextX+1
            nextY = nextY+1
        # check left down
        elif(skeleton[nextY+1][nextX-1]):
            nextX = nextX-1
            nextY = nextY+1
        # if none exit loop
        else:
            break
    if(len(pathR) >= 5 and count >= 2 and countD >= 4):
        isCorner.append([xD, yD])
        for i in pathR:
            pX = i[0]
            pY = i[1]
            visited[pY][pX] = 1


def findVinverted(skeleton):
    h, w = skeleton.shape
    visited = np.zeros((h, w))
    isCorner = []
    index = 0
    pixelIndices = np.where(skeleton == 1)
    pixelCount = len(pixelIndices[0])
    for i in range(pixelCount):
        y = pixelIndices[0][i]
        x = pixelIndices[1][i]
        # check if in white pixel is on borders
        if(x+1 == w or y+1 == h or x == 0 or y == 0):
            continue
        downR = skeleton[y+1][x+1]
        down = skeleton[y+1][x]
        right = skeleton[y][x+1]
        downL = skeleton[y+1][x-1]
        if (downR and downL):
            checkPathRightDown(y+1, x+1, y+1, x-1, visited, isCorner, skeleton)
        elif (downR and down):
            checkPathRightDown(y+1, x+1, y+1, x, visited, isCorner, skeleton)
        elif (right and downL):
            checkPathRightDown(y, x+1, y+1, x-1, visited, isCorner, skeleton)
        elif (right and down):
            checkPathRightDown(y, x+1, y+1, x, visited, isCorner, skeleton)
    return len(isCorner)

# if color is white => symbol , black => background
# another matrix => visited pixels

# attempt algorithm to find corner
# 1.find suspected point as a begining of corner
# 2.iterate from the right pixel
# 3.if path from this pixel is longer than 4 and dominating direction is Up AND RIGHT and does not form a loop(closed path)
#  then this is a corner=> mark as visited
# 4.repeat untill all white pixels are done

# return path length and top two dominating directions


def checkPathRightUp(yR, xR, yU, xU, visited, isCorner, skeleton):
    # if valid path update visited and isCorner list
    h, w = visited.shape
    if(visited[yR][xR]):
        return
    nextX = xR
    nextY = yR
    pathR = []
    pathR.append([nextX, nextY])
    count = 0
    while(nextX+1 != w and nextY != 0):
        # check right pixel
        if(skeleton[nextY][nextX+1]):
            nextX = nextX+1
            pathR.append([nextX, nextY])
        # check right up
        elif(skeleton[nextY-1][nextX+1]):
            nextX = nextX+1
            nextY = nextY-1
            pathR.append([nextX, nextY])
            count += 1
        # check up
        elif(skeleton[nextY-1][nextX]):
            nextY = nextY-1
            pathR.append([nextX, nextY])
            count += 1
        # if none exit loop
        else:
            break
    nextX = xU
    nextY = yU
    countU = 0
    while(nextX+1 != w and nextY != 0 and nextX != 0):
        # check up
        if(skeleton[nextY-1][nextX]):
            nextY = nextY-1
            countU += 1
        # check left up pixel
        elif(skeleton[nextY-1][nextX-1]):
            nextX = nextX-1
            nextY = nextY-1
        # check right up
        elif(skeleton[nextY-1][nextX+1]):
            nextX = nextX+1
            nextY = nextY-1
        # if none exit loop
        else:
            break
    if(len(pathR) >= 5 and count >= 2 and countU >= 4):
        isCorner.append([xR, yR])
        for i in pathR:
            pX = i[0]
            pY = i[1]
            visited[pY][pX] = 1


def findV(skeleton):
    h, w = skeleton.shape
    visited = np.zeros((h, w))
    isCorner = []
    index = 0
    pixelIndices = np.where(skeleton == 1)
    pixelCount = len(pixelIndices[0])
    for i in range(pixelCount):
        y = pixelIndices[0][i]
        x = pixelIndices[1][i]
        # check if in white pixel is on borders
        if(x+1 == w or y+1 == h or x == 0 or y == 0):
            continue
        upR = skeleton[y-1][x+1]
        up = skeleton[y-1][x]
        right = skeleton[y][x+1]
        upL = skeleton[y-1][x-1]
        if (upR and upL):
            checkPathRightUp(y-1, x+1, y-1, x-1, visited, isCorner, skeleton)
        elif (upR and up):
            checkPathRightUp(y-1, x+1, y-1, x, visited, isCorner, skeleton)
        elif (right and up):
            checkPathRightUp(y, x+1, y-1, x, visited, isCorner, skeleton)
        elif (right and upL):
            checkPathRightUp(y, x+1, y-1, x-1, visited, isCorner, skeleton)
    return len(isCorner)
