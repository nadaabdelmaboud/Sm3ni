from preprocessing import *
from staffremoval import *
from digitsClassifier import *


BinarizedImage = preprocessing("01.PNG")
staffRemoval(BinarizedImage)


#digits classifier take care to send image with black background
predictednum = digitsClassifier("test2.jpg")
print(predictednum)