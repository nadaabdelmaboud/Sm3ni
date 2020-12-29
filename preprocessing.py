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
        rotated = inter.rotate(img, trueAngle, reshape=True, order=0)
    return rotated

def feng_threshold(img, w_size1=15, w_size2=30,
                   k1=0.15, k2=0.01, alpha1=0.1):
    """ Runs the Feng's thresholding algorithm.
    Reference:
    Algorithm proposed in: Meng-Ling Feng and Yap-Peng Tan, “Contrast adaptive
    thresholding of low quality document images”, IEICE Electron. Express,
    Vol. 1, No. 16, pp.501-506, (2004).
    Modifications: Using integral images to compute the local mean and the
    standard deviation
    @param img: The input image. Must be a gray scale image
    @type img: ndarray
    @param w_size1: The size of the primary local window to compute
        each pixel threshold. Should be an odd window
    @type w_size1: int
    @param w_size2: The size of the secondary local window to compute
        the dynamic range standard deviation. Should be an odd window
    @type w_size2: int
    @param k1: Parameter value that lies in the interval [0.15, 0.25].
    @type k1: float
    @param k2: Parameter value that lies in the interval [0.01, 0.05].
    @type k2: float
    @param alpha1: Parameter value that lies in the interval [0.15, 0.25].
    @type alpha1: float
    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size1 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Obtaining windows
    padded_img = np.ones((rows + w_size1 - 1, cols + w_size1 - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size1, w_size1))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))

    # Obtaining local coordinates for std range calculations
    hw_size = w_size2 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 2) * (x2 - x1 + 2)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means2
    means2 = sums / l_size

    # Computing standard deviation range
    std_ranges = np.sqrt(sqr_sums / l_size - np.square(means2))

    # Computing normalized standard deviations and extra alpha parameters
    n_stds = stds / std_ranges
    n_sqr_std = np.square(n_stds)
    alpha2 = k1 * n_sqr_std
    alpha3 = k2 * n_sqr_std

    thresholds = ((1 - alpha1) * means + alpha2 * n_stds
                  * (means - mins) + alpha3 * mins)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(img[x][y]<thresholds[x][y]):
                img[x][y]=0
            else:
                img[x][y]=1
    return img

#Smoothing
def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=0)
    return np.where(sd == 0, 0, m/sd)
def smooth(gray_img):   
    snr = signaltonoise(gray_img)
    if(snr<1):
        snr=1
    if(snr>5):
        snr=5
    sigma=(-50/4)*(snr-1)+60
    smImage=cv2.bilateralFilter(gray_img,8,int(sigma),int(sigma))
    return smImage


#Illumenation
def simplestColorBalance(img,s):
    s/=100
    n,m=img.shape
    f=img.flatten()
    f.sort()
    minT=f[int(s*n*m)]
    maxT=f[ n*m - int(s*n*m) - 1]
    img=np.where(img<minT,0,img)
    img=np.where(img>maxT,255,img)
    img=np.where((img!=0)&(img!=255),255*((img-minT)/(maxT-minT)),img)
    return img
def poisonScreening(img,L):
    n,m=img.shape
    img_freq=cv2.dft(img)
    for i in range(n):
        for j in range(m):
            coef=(pi*pi*i*i)/(n*n) + (pi*pi*j*j)/(m*m)
            img_freq[i][j]=(img_freq[i][j]*coef)/(L+coef)
    img=cv2.idft(img_freq)
    return img
def applyPoison(img,s,L):
    img=simplestColorBalance(img,s)
    img=poisonScreening(img,L)
    img=simplestColorBalance(img,s)
    return img
def prespectiveCorrection(RotatedImage):
    #rotated image is the image returned from the deskew directly
    RotatedImage = RotatedImage*255
    rows,cols = RotatedImage.shape
    edges = cv2.Canny(RotatedImage,50,150,apertureSize = 3)
    
    maxLineGap = 10
    minLineLength = 60
    prespectiveSlopeThreshold = 0.05
    flag=False
    
    
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    binary = np.zeros((rows,cols)).astype('uint8')
    
    s = 0
    countLines = 0
    countOfVerticalAndhorizontalLines = 0
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(binary,(x1,y1),(x2,y2),255,2)
            m = ((y2-y1)/(x2-x1))
            if(abs(m)==np.inf or m==0):
                countOfVerticalAndhorizontalLines+=1
            if(abs(m) != np.inf):
                countLines+=1
                s+=m
                
    avrgSlope = s/countLines 
    percentage = (countOfVerticalAndhorizontalLines/len(lines))*100
              

    if(abs(avrgSlope)>prespectiveSlopeThreshold and percentage<80):
        
        SE = np.ones([50,50])      
        closedImage = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, SE)
        
    
        contours, hierarchy = cv2.findContours(closedImage,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
        
        for box in cnts:
            epsilon = 0.02*cv2.arcLength(box,True)
            approx = cv2.approxPolyDP(box,epsilon,True) 
            for i in range(0,len(approx)):
                RotatedImage = cv2.circle(RotatedImage, (approx[i,0,0],approx[i,0,1]), 2, 200, 2)
            if(len(approx)==4):
                flag=True
                points=approx
                break
            
        if flag==True:
            
            margin = 20
            
            points[0,0,0]+=margin
            points[0,0,1]-=margin
            points[1]-=margin
            points[2,0,0]-=margin
            points[2,0,1]+=margin
            points[3]+=margin
            
            
            cv2.drawContours(RotatedImage, [points], -1, (0, 255, 0), 2)
            
            rect = np.zeros((4, 2), dtype = "float32")
            s1 = points.sum(axis = 1)
            s = s1.sum(axis=1)
            
            
            rect[0] = s1[np.argmin(s)]
            rect[2] = s1[np.argmax(s)]
            diff = np.diff(s1, axis = 1)
            rect[1] = s1[np.argmin(diff)]
            rect[3] = s1[np.argmax(diff)]
            
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")
                
        
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(RotatedImage, M, (maxWidth, maxHeight)) 
            
        else:warped = RotatedImage
            
    else:
        warped = RotatedImage
        
    return warped
def preprocessing(imgPath):
    img= cv2.imread(imgPath) 
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    IlluminatedImage=applyPoison(gray_img,0.1,0.1)
    IlluminatedImage=IlluminatedImage.astype(np.uint8)
    smoothedImage=smooth(IlluminatedImage)
    BinarizedImage=feng_threshold(smoothedImage)
    BinarizedImage=1-BinarizedImage
    RotatedImage=deskew(BinarizedImage)
    correctedImg=prespectiveCorrection(RotatedImage)
    BinarizedImage=1-correctedImg
    cv2.imwrite("Binarized.png",BinarizedImage*255)
    return BinarizedImage