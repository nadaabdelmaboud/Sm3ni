from preprocessing.preprocessing import *
from staffremoval.staffremoval import *
from classifiers.digitsClassifier import *
from classifiers.accedintalsClassifier import *
from inout.generateOutput import *
from features.checkScanned import *
from features.extractfeatures import *
import pickle
import os
import cv2
import numpy as np
import argparse
import os
import datetime
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help="Input File")
parser.add_argument("outputfolder", help="Output File")

args = parser.parse_args()
fileExists = False 
inputFolder = args.inputfolder
outputFolder = args.outputfolder

digitsFileModel = 'models/digits_model.sav'
symbolsFileModel = 'models/symbols_model.sav'
accedintalsFileModel = 'models/accedintals_model.sav'

loaded_digits_model = pickle.load(open(digitsFileModel, 'rb'))
loaded_symbols_model = pickle.load(open(symbolsFileModel, 'rb'))
loaded_accedintals_model = pickle.load(open(accedintalsFileModel, 'rb'))

for fNum, filename in enumerate(os.listdir(inputFolder)):
    binarizedImg = preprocessing(inputFolder + '/' + filename)
    isHorizontal = getHorizontalLines(binarizedImg)
    if(isHorizontal):
        segContours, segContoursDim, maxSpace, checkNumList, segPeakMids, segWidths = staffRemoval(binarizedImg)
    else:
        segContours, segContoursDim, maxSpace, checkNumList, segPeakMids, segWidths, segAspects ,widths,heights,Ys= staffRemovalNonHorizontal(binarizedImg)

    outFileName = filename.split('.')[0]
    f = open(outputFolder + '/g' + outFileName+'.txt', "w")

    if(len(segContours) > 1):
        f.write("{\n")

    for i, seg in enumerate(segContours):
        nums = []
        f.write("[ ")
        hasAccidental = False
        accidental = ""
        for j, image in enumerate(seg):
            if checkNumList[i][j] == 1:
                #cv2.imwrite("num"+str(i)+"_"+str(j)+".png",image*255)
                features = extractDigitsFeatures(image)
                result = loaded_digits_model.predict([features])
                c = result[0]
                if (c == 'b'):
                    nums.append(2)
                else:
                    nums.append(4)
                if(len(nums) == 2):
                    lineOut = '\meter<"' + str(nums[0])+'/'+str(nums[1])+'">'
                    f.write(lineOut)
            else:
                if(isHorizontal):
                    features, Bblobs, Wblobs = extractFeatures(image, maxSpace)
                else:
                    features, Bblobs, Wblobs = extractFeatures(image, maxSpace,segAspects[i][j],widths[i][j],heights[i][j],Ys[i][j])

                if((len(Bblobs)+len(Wblobs)) > 0):
                    print("f : ",i,j)
                    print(features)
                    #cv2.imwrite(str(i)+"_"+str(j)+".png",image*255)
                    ClassifierVote = loaded_symbols_model.predict([features])[0]
                    if(isHorizontal):
                        className, Notes, duration = NoteOut(ClassifierVote, Bblobs, Wblobs, segContoursDim[i][j][2], segContoursDim[i][j][0], segPeakMids[i], segWidths[i])
                    else:
                        className, Notes, duration = NoteOut(ClassifierVote, Bblobs, Wblobs, segContoursDim[i][j][2], segContoursDim[i][j][0], segPeakMids[i][j], segWidths[i][j],maxSpace)
                  
                    if(hasAccidental):
                        lineOut = formatLine(className,Notes,duration,accidental)
                        hasAccidental=False
                    else:
                        lineOut = formatLine(className, Notes, duration, '')
                    f.write(lineOut)
                else:
                    # call accidentals classifier
                    #cv2.imwrite("acc"+str(i)+"_"+str(j)+".png",image*255)
                    hasAccidental = True
                    features = extractAccedintalsFeatures(image)
                    result = loaded_accedintals_model.predict([features])
                    accidental = getAccedintals(result)
                    if(accidental == 'clef' or accidental == 'bar'):
                        hasAccidental = False
                    elif(accidental == '.'):
                        w = segContoursDim[i][j][1] - segContoursDim[i][j][0]
                        h = segContoursDim[i][j][3] - segContoursDim[i][j][2]
                        if(h/w > 1.2 or h/w < 0.8):
                            hasAccidental = False
           
        if i == len(segContours)-1:
            f.write("]")
        else:
            f.write("],")
        if j == len(seg)-1:
            f.write("\n")
    if len(segContours) > 1:
        f.write("\n}")
    f.close()