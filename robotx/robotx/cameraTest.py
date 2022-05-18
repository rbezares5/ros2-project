import numpy as np
import cv2 as cv
import copy

def createMaskR(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.062
    hMax=0.141
    sMin=0.365
    sMax=1
    vMin=0.541
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask

def createMaskB(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.544
    hMax=0.714
    sMin=0.262
    sMax=1
    vMin=0.262
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask

def createMaskP(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.819
    hMax=1
    sMin=0.365
    sMax=1
    vMin=0.631
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask

def createMaskG(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.163
    hMax=0.409
    sMin=0.082
    sMax=1
    vMin=0.498
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask        



cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#firstly, we are going to just get a roi of the board
ret, frame = cap.read()
r=cv.selectROI(frame)
#imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#cv.imshow("Image", imCrop)

#Now we are going to detect the coordinates of the centroids of the squares of the board

print('Detecting empty board. Press q to continue')
while True:
    ret, frame = cap.read()
    frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Detect chessboard squares using canny!
    cv.imshow('grayscale', gray)
    #bwFrame=copy.deepcopy(gray)
    #bwframe
    grayBlur = cv.GaussianBlur(gray, (15, 15), 2)
    cv.imshow('gray blur', grayBlur)
    edges=cv.Canny(grayBlur,10,100)
    cv.imshow('edges', edges)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    edges=cv.dilate(edges, kernel)
    cv.imshow('dilated edges', edges)
    edges=cv.bitwise_not(edges)
    cv.imshow('inverted edges', edges)

    # find contours
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # filter out contours by size
    squares = []
    for con in contours:
        area = cv.contourArea(con)
        if (area < 1000) & (area > 200): # size threshold
            squares.append(con)
    #cv.drawContours(edges, squares, -1, (0), -1)
    cv.imshow('filtered squares', edges)

    #find centroids using moments
    i=1
    centroidsSquares=[]
    for c in squares:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroidsSquares.append([cX,cY])
        #cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        #cv.putText(frame, str(i), (cX, cY),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+=1
    #cv.imshow('Squares centroids', frame)

    # WE HAVE TO ORDER THE CENTROIDS OF THE SQUARES
    if len(centroidsSquares) > 0:
        centroidsSquares=np.asarray(centroidsSquares)
        #print(centroidsSquares)
        _, bins =np.histogram(centroidsSquares[:,1], bins='auto')
        bins=bins+10 #we shift the bins edges to account for approximation error
        #print(bins)
        centroidsSquaresOrder=np.digitize(centroidsSquares[:,1],bins,right=False)
        #print(centroidsSquaresOrder)
        centroidsSquares2=copy.deepcopy(centroidsSquares)
        centroidsSquares2[:,1]=centroidsSquaresOrder
        #print(centroidsSquares2)
        index=np.lexsort((centroidsSquares2[:,0],centroidsSquares2[:,1]))
        centroidsSquares3=[(centroidsSquares[i,0],centroidsSquares[i,1]) for i in index]
        centroidsSquares3=np.asarray(centroidsSquares3)
        #print(centroidsSquares3)

        
        i=1
        for j in range(len(centroidsSquares3)):
            cX=centroidsSquares3[j,0]
            cY=centroidsSquares3[j,1]
            cv.putText(frame, str(i), (cX, cY),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            i+=1
        cv.imshow('Squares centroids', frame)

    if len(centroidsSquares3) == 64:
        print('SQUARES DETECTED SUCCESFULLY')

        #turn CentroidsSquares3 into a reduced list of 32, discarding non-playable squares
        centroidsSquares4=np.zeros((32,2), dtype=int)
        k=0
        for i in range(8):
            for j in range(8):
                if (i+j+1)%2==1:
                    #print(k)
                    centroidsSquares4[k]=centroidsSquares3[np.ravel_multi_index((i,j),(8,8))]
                    k+=1

        i=1
        for j in range(len(centroidsSquares4)):
            cX=centroidsSquares4[j,0]
            cY=centroidsSquares4[j,1]
            cv.putText(frame, str(i), (cX, cY),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            i+=1
        cv.imshow('Squares centroids', frame)

        #print(centroidsSquares3)
        #print(centroidsSquares4)

        break   #finish this section

    if cv.waitKey(1) == ord('q'):
        break   

#MAIN LOOP
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    #crop the image using the ROI
    frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(frame, (15, 15), 2)
    cv.imshow('color blur',blur)
    #color segmentation
    maskR=createMaskR(blur)
    cv.imshow('Red mask', maskR)

    maskB=createMaskB(blur)
    cv.imshow('Blue mask', maskB)

    maskP=createMaskP(blur)
    cv.imshow('Pink mask', maskP)
    maskG=createMaskG(blur)
    cv.imshow('Green mask', maskG)

    #morphological operations to get only the pieces
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

    #imgR=cv.morphologyEx(maskR, cv.MORPH_OPEN, kernel)
    imgR=cv.morphologyEx(maskR, cv.MORPH_CLOSE, kernel)
    contoursR, hierarchyR=cv.findContours(imgR, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #print(len(contoursR))
    cv.imshow('Red pieces', imgR)
    frameR=copy.deepcopy(frame)
    #find centroids using moments
    i=1
    centroidsR=[]
    for c in contoursR:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroidsR.append([cX,cY])
        cv.circle(frameR, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frameR, str(i), (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+=1
    cv.imshow('Red centroids', frameR)


    imgB=cv.morphologyEx(maskB, cv.MORPH_ERODE, kernel)
    imgB=cv.morphologyEx(imgB, cv.MORPH_CLOSE, kernel)
    contoursB, hierarchyB=cv.findContours(imgB, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    #print(len(contoursB))
    cv.imshow('Blue pieces', imgB)
    frameB=copy.deepcopy(frame)
    #find centroids using moments
    i=1
    centroidsB=[]
    for c in contoursB:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroidsB.append([cX,cY])
        cv.circle(frameB, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frameB, str(i), (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+=1
    cv.imshow('Blue centroids', frameB)

    imgP=cv.morphologyEx(maskP, cv.MORPH_CLOSE, kernel)
    contoursP, _ =cv.findContours(imgP, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.imshow('Pink pieces', imgP)
    frameP=copy.deepcopy(frame)
    #find centroids using moments
    i=1
    centroidsP=[]
    for c in contoursP:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroidsP.append([cX,cY])
        cv.circle(frameP, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frameP, str(i), (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+=1
    cv.imshow('Pink centroids', frameP)
    
    imgG=cv.morphologyEx(maskG, cv.MORPH_ERODE, kernel)
    imgG=cv.morphologyEx(imgG, cv.MORPH_CLOSE, kernel)
    contoursG, _ =cv.findContours(imgG, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    cv.imshow('Green pieces', imgG)
    frameG=copy.deepcopy(frame)
    #find centroids using moments
    i=1
    centroidsG=[]
    for c in contoursG:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroidsG.append([cX,cY])
        cv.circle(frameG, (cX, cY), 5, (255, 255, 255), -1)
        cv.putText(frameG, str(i), (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        i+=1
    cv.imshow('Green centroids', frameG)


    
    '''
    #harris corner detector
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    cv.imshow('dst', dst)
    # Threshold for an optimal value, it may vary depending on the image.
    harris=copy.deepcopy(frame)
    harris[dst>0.01*dst.max()]=[0,0,255]   
    # Display the resulting frame
    cv.imshow('harris corners', harris)
    '''


    

    #Now we compare the coordinates of each piece with the coordinates of each square
    boardState=np.zeros((8,4), dtype=int)

    if len(centroidsR)>0:
        for i in range(len(centroidsR)):
            pos=0
            minDist=1000
            for j in range(len(centroidsSquares4)):
                p1=np.array([centroidsSquares4[j,0], centroidsSquares4[j,1]]) 
                p2=np.array([centroidsR[i][0], centroidsR[i][1]])
                dist=p1-p2
                dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                if dist < minDist:
                    minDist=dist
                    pos=j
            boardState[np.unravel_index(pos,(8,4))]=1 #linear index

    if len(centroidsB)>0:
        for i in range(len(centroidsB)):
            pos=0
            minDist=1000
            for j in range(len(centroidsSquares4)):
                p1=np.array([centroidsSquares4[j,0], centroidsSquares4[j,1]]) 
                p2=np.array([centroidsB[i][0], centroidsB[i][1]])
                dist=p1-p2
                dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                if dist < minDist:
                    minDist=dist
                    pos=j
            boardState[np.unravel_index(pos,(8,4))]=2 #linear index

    if len(centroidsP)>0:
        for i in range(len(centroidsP)):
            pos=0
            minDist=1000
            for j in range(len(centroidsSquares4)):
                p1=np.array([centroidsSquares4[j,0], centroidsSquares4[j,1]]) 
                p2=np.array([centroidsP[i][0], centroidsP[i][1]])
                dist=p1-p2
                dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                if dist < minDist:
                    minDist=dist
                    pos=j
            boardState[np.unravel_index(pos,(8,4))]=3 #linear index

    if len(centroidsG)>0:
        for i in range(len(centroidsG)):
            pos=0
            minDist=1000
            for j in range(len(centroidsSquares4)):
                p1=np.array([centroidsSquares4[j,0], centroidsSquares4[j,1]]) 
                p2=np.array([centroidsG[i][0], centroidsG[i][1]])
                dist=p1-p2
                dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                if dist < minDist:
                    minDist=dist
                    pos=j
            boardState[np.unravel_index(pos,(8,4))]=4 #linear index

    #print(boardState)
    #input('press enter to continue')

    # Finish the program
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()