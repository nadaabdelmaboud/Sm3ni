import numpy as np
import cv2


def getNoteName(row, col, peaksMids, width,maxSpace):

    segment = peaksMids
    widths = width
    spaces = []
    for i in range(segment.shape[0]):
        if(i == 0):
            continue
        spaces.append(
            int(((segment[i]-segment[i-1])-int(widths[i]/2+widths[i-1]/2))))
    spaces = np.array(spaces)
    note = ''
    threshold = 3
    segment = list(segment)
    spaces = list(spaces)
    widths = list(widths)
    while len(segment) < 5:
        segment.append(segment[len(segment)-1]+maxSpace)

    while len(spaces) < 4:
        spaces.append(maxSpace)

    while len(widths) < 5:
        widths.append(widths[len(widths)-1])

    if(row == segment[0] or (row <= segment[0]+threshold and row >= segment[0]-threshold)):
        note = "f2"
    if(row == segment[1] or (row <= segment[1]+threshold and row >= segment[1]-threshold)):
        note = "d2"
    if(row == segment[2] or (row <= segment[2]+threshold and row >= segment[2]-threshold)):
        note = "b1"
    if(row == segment[3] or (row <= segment[3]+threshold and row >= segment[3]-threshold)):
        note = "g1"
    if(row == segment[4] or (row <= segment[4]+threshold and row >= segment[4]-threshold)):
        note = "e1"
    if(note == ''):
        if(row > segment[0] and row < segment[1]):
            note = "e2"
        if(row > segment[1] and row < segment[2]):
            note = "c2"
        if(row > segment[2] and row < segment[3]):
            note = "a1"
        if(row > segment[3] and row < segment[4]):
            note = "f1"
    if(note == ''):
        line6 = segment[4]+spaces[3]+widths[4]
        line0 = segment[0]-spaces[0]-widths[0]
        linemin1 = line0-spaces[0]-widths[0]
        if(row >= line6-threshold):
            note = 'c1'
        if(row == line0 or (row <= line0+threshold and row >= line0-threshold)):
            note = 'a2'
        if(note == ''):
            if(row < segment[0] and row > line0):
                note = "g2"
            if(row < line0 and row > linemin1):
                note = "b2"
            if(row > segment[4] and row < line6):
                note = 'd1'
    return note


def NoteOut(classifierVote, Bblobs, Wblobs, Ymin, Xmin, peaksMids, widths,maxSpace=0):

    BlobsCenters = []
    NoteName = ''
    if(len(Bblobs) > 0):
        BlobsCenters = Bblobs
    elif(len(Wblobs) > 0):
        BlobsCenters = Wblobs
    className = classifierVote
    duration = '4'
    if(len(className) == 3 or len(className) == 2):
        if(className == 'be8'):
            duration = '8'
        else:
            duration = className[1:]
    elif(len(className) == 4):
        duration = className[2:]
    Notes = []
    if(className[0:2] == "be"):
        BlobsCenters = sorted(BlobsCenters, key=lambda x: x[1])
    for center in BlobsCenters:
        centerx = int(center[0]+Ymin)
        centery = int(center[1]+Xmin)
        NoteName = getNoteName(centerx, centery, peaksMids, widths,maxSpace)
        Notes.append(NoteName)
    return className, Notes, duration


def formatLine(className, Notes, duration, accidental):
    outLine = ''
    if(className == 'c'):
        Notes = sorted(Notes, key=str.lower)
        outLine = ' {'
        for note in Notes:
            if(len(note)>=2):
                if(accidental == '.'):
                    outLine += accidental+' '+note[0]+note[1]+'/'+str(duration)+','
                else:
                    outLine += note[0]+accidental+note[1]+'/'+str(duration)+','
        outLine = outLine[0:len(outLine)-1]
        outLine += '}'
    elif(className[0:2] == "be"):
        for note in Notes:
            if(len(note)>=2):
                if(accidental == '.'):
                    outLine += accidental+' '+note[0]+note[1]+'/'+str(duration)
                else:
                    outLine += ' '+note[0]+accidental+note[1]+'/'+str(duration)

    elif len(Notes) > 0:
        if(len(Notes[0])>=2):
            if(accidental == '.'):
                outLine += accidental+' ' + \
                    Notes[0][0]+Notes[0][1]+'/'+str(duration)
            else:
                outLine += ' '+Notes[0][0]+accidental+Notes[0][1]+'/'+str(duration)

    return outLine
