import numpy as np
from scipy.io.wavfile import write


#CREDITS TO How to Play Music Using Mathematics in Python Articale WRITTEN BY Nishu Jain
#https://towardsdatascience.com/mathematics-of-music-in-python-b7d838c84f72

noteFreq = {'c1' : 261.6 , 'c#1': 277.2 , 'c2' : 523.3 , 'c#2' : 554.4 ,
            'd1' : 293.7 , 'd#1': 311.1 , 'd2' : 587.3 , 'd#2' : 622.3 ,
            'e1' : 329.6 , 'e#1': 329.6 , 'e2' : 659.3 , 'e#2' : 659.3 ,
            'f1' : 349.2 , 'f#1': 370.0 , 'f2' : 698.5 , 'f#2' : 740.0 ,
            'g1' : 392.0 , 'g#1': 415.3 , 'g2' : 784.0 , 'g#2' : 830.6 ,
            'a1' : 440.0 , 'a#1': 466.2 , 'a2' : 880.0 , 'a#2' : 932.3 ,
            'b1' : 493.9 , 'b#1': 493.9 , 'b2' : 987.8 , 'b#2' : 987.8}


sampleRate = 44100

def getWave(duration , freq):
    
    A = 4096
    t = np.linspace(0, duration, int(sampleRate * duration))
    wave = A * np.sin(2 * np.pi * freq * t)
    return wave



def getAudio(notes,durations):
    Audio = []
    for i in range (len(notes)):
        noteWave = getWave(durations[i],noteFreq[notes[i]])
        Audio.append(noteWave)
    Audio = np.concatenate(Audio)
    return Audio
    
def readingNotesFromFile(filePath):
    f = open(filePath , "r")
    s = f.read()
    f.close()
    return s



def parseString(notesString):

    notesString = notesString.lower()
    notesString.replace("\n", " ")
    
    notesString = notesString.split()
    
    durations = []
    notes = []
    for i in notesString:
        i = i.split(']')[0]
        if i == '' or i == '[' or i.find('<') != -1 or i.find('>') != -1 or len(i)==1:
            continue
        if i[0] == '{':
            i = i.split('{')[1]
            i = i.split('}')[0]
            parseBeamOrClif = i.split(',')
            for j in parseBeamOrClif:
                n = j.split('/')[0]
                
                if len(n)>1 and n[1] == '&':
                    n = n[:1]+'#'+n[2:]
                    
                if len(n) == 4 and (n[1]=='#' or n[1] == '&') and (n[2]=='#' or n[2] == '&'):
                    n = n[:1] + '#' + n[3:]
                
                
                d = j.split('/')[1]
                if d == '' or d ==' ':
                    d = '2'
                d = int(d)
                
                notes.append(n)
                durations.append(1/d)
                
        else:
            n = i.split('/')[0]
            if len(n)>1 and n[1] == '&':
                n = n[:1]+'#'+n[2:]
                
            if len(n) == 4 and (n[1]=='#' or n[1] == '&') and (n[2]=='#' or n[2] == '&'):
                n = n[:1] + '#' + n[3:]
                
            
            d = i.split('/')[1]
            d = d.split('.')[0]
            if d == '' or d ==' ':
                d = '2'
            d = int(d)
            
            notes.append(n)
            durations.append(1/d)  
    
    return notes,durations

def generateAudio(inputTextFilePath,OutputAudioPath = '',audioName = 'audio'):
    notes = []
    durations = []
    notesString = readingNotesFromFile(inputTextFilePath)
    notes , durations = parseString(notesString)


    if notes != []:  
        Audio = getAudio(notes,durations)

        Audio = Audio * (16300/np.max(Audio)) # Adjusting the Amplitude
        if OutputAudioPath == '':
            write(audioName +'.wav', sampleRate, Audio.astype(np.int16))
        else:
            write(OutputAudioPath + '/' + audioName +'.wav', sampleRate, Audio.astype(np.int16))
          
            




