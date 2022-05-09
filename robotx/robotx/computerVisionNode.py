import rclpy
from rclpy.node import Node

import numpy as np
import cv2 as cv
#import time

# Import custom interfaces
from interfaces.srv import ComputerVision

# Define color segmentation functions
# some thresholds have been modified to work better in lab
def createMaskR(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.062
    hMax=0.235
    sMin=0.348
    sMax=1
    vMin=0.597
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
    sMin=0.318
    sMax=1
    vMin=0.356
    vMax=0.781
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
    hMin=0.153
    hMax=0.404
    sMin=0.159
    sMax=1
    vMin=0.614
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask

# Operation functions that we can repeat for each color
def findCentroids(mask,kernel):
    #maskImprove = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    maskImprove = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(maskImprove, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # find centroids using moments
    i=1
    centroids=[]
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)

        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroids.append([cX,cY])
        i+=1

    return centroids 

def populateBoard(board,squares,pieces,symbol):
    for i in range(len(pieces)):
        pos=0
        minDist=1000
        for j in range(len(squares)):
            p1=np.array([squares[j,0],squares[j,1]])
            p2=np.array([pieces[i,0],pieces[i,1]])
            dist=p1-p2
            dist=(dist[0]**2 + dist[1]**2)**(0.5)
            if dist < minDist:
                            minDist=dist
                            pos=j

        board[np.unravel_index(pos,(8,4))]=symbol #use linear indexing to assign the symbol to the right position
        #TODO
        #here we could also store the coordinates of the pieces using the same structure as the logical board in order to get more accurate info for the robot
    
    return board

class computerVisionNode(Node):

    def __init__(self,pCamera):
        super().__init__('computer_vision_server') #initilize the node with this name
        self.srv = self.create_service(ComputerVision, 'computer_vision_service', self.computerVisionCallback) #type, name and callback of the service
        self.cap=pCamera

    def computerVisionCallback(self, request, response):
        self.get_logger().info('Vision request acknowledged')   #receive the request

        if request.request:
            # Here we process the image and finally create a matrix representing the board state

            # COMPUTER VISION CODE
            # create camera object
            '''
            print('Opening camera')
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            '''
            

            # there's a weird glitch that I only get in the lab, but I can't detect anything using a single frame
            # it seems to work normal if I just take several pics, so I put a while loop here
            i=0
            while i<5:
                # Capture frame-by-frame
                #print('Acquiring image')
                _, frame = self.cap.read()
                #cv.imshow('frame',frame)
                #time.sleep(1)

                # Our operations on the frame come here
                #crop the image using the ROI
                r=request.roi
                frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                blur = cv.GaussianBlur(frame, (15, 15), 2)

                #print('Separating colors')
                #time.sleep(0.5)
                #color segmentation
                maskR=createMaskR(blur)
                maskB=createMaskB(blur)
                maskP=createMaskP(blur)
                maskG=createMaskG(blur)

                #cv.imshow('Red mask', maskR)
                #cv.imshow('Blue mask', maskB)
                #cv.imshow('Pink mask', maskP)
                #cv.imshow('Green mask', maskG)

                #print('Performing morphological operations')
                #time.sleep(0.5)
                #morphological operations to get only the pieces
                kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

                # Perform the same operations for each color to get respective centroids
                centroidsR=findCentroids(maskR,kernel)
                #print(len(centroidsR))
                centroidsB=findCentroids(maskB,kernel)
                #print(len(centroidsB))
                centroidsP=findCentroids(maskP,kernel)
                #print(len(centroidsP))
                centroidsG=findCentroids(maskG,kernel)
                #print(len(centroidsG))
                '''
                imgR = cv.morphologyEx(maskR, cv.MORPH_CLOSE, kernel)
                contoursR, _ = cv.findContours(imgR, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                frameR = copy.deepcopy(frame)
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

                #repeat the same operations for blue pieces
                imgB = cv.morphologyEx(maskB, cv.MORPH_CLOSE, kernel)
                contoursB, _ = cv.findContours(imgB, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                frameB = copy.deepcopy(frame)
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

                #repeat the same operations for pink pieces
                '''

                # Now we compare the coordinates of each piece with the coordinates of each square
                # We will need an empty board and the coordinates for the playabale centroids (request data)
                boardState = np.zeros((8,4), dtype=int)
                centroidsSquares = request.boardcoords
                centroidsSquares = np.reshape(centroidsSquares, (32,2))

                # Modify the board adding the pieces of each color
                if len(centroidsR)>0:
                    centroidsR=np.asarray(centroidsR)
                    print('red coordinates')
                    print(len(centroidsR))
                    print(centroidsR)
                    boardState=populateBoard(boardState,centroidsSquares,centroidsR,symbol=1)
                
                if len(centroidsB)>0:
                    centroidsB=np.asarray(centroidsB)
                    print('blue coordinates')
                    print(len(centroidsB))
                    print(centroidsB)
                    boardState=populateBoard(boardState,centroidsSquares,centroidsB,symbol=2)

                if len(centroidsP)>0:
                    centroidsP=np.asarray(centroidsP)
                    print('pink coordinates')
                    print(len(centroidsP))
                    print(centroidsP)
                    boardState=populateBoard(boardState,centroidsSquares,centroidsP,symbol=3)

                if len(centroidsG)>0:
                    centroidsG=np.asarray(centroidsG)
                    print('green coordinates')
                    print(len(centroidsG))
                    print(centroidsG)
                    #boardState=populateBoard(boardState,centroidsSquares,centroidsG,symbol=4)

                i=i+1

            '''
            #red
            centroidsR = np.asarray(centroidsR)
            if len(centroidsR)>0:
                for i in range(len(centroidsR)):
                    pos=0
                    minDist=1000
                    for j in range(len(centroidsSquares)):
                        p1=np.array([centroidsSquares[j,0], centroidsSquares[j,1]]) 
                        p2=np.array([centroidsR[i,0], centroidsR[i,1]])
                        dist=p1-p2
                        dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                        if dist < minDist:
                            minDist=dist
                            pos=j
                    boardState[np.unravel_index(pos,(8,4))]=1 #linear index

            #blue
            centroidsB = np.asarray(centroidsB)
            if len(centroidsB)>0:
                for i in range(len(centroidsB)):
                    pos=0
                    minDist=1000
                    for j in range(len(centroidsSquares)):
                        p1=np.array([centroidsSquares[j,0], centroidsSquares[j,1]]) 
                        p2=np.array([centroidsB[i,0], centroidsB[i,1]])
                        dist=p1-p2
                        dist=(dist[0]**2 + dist[1]**2)**(0.5) 
                        if dist < minDist:
                            minDist=dist
                            pos=j
                    boardState[np.unravel_index(pos,(8,4))]=2 #linear index
            '''

            # Convert the array into a list so that it can be passed
            boardState=boardState.flatten() 
            boardState=boardState.tolist()

            response.board=boardState
            response.goal = True

        else:
            response.goal = False
        
        return response


def main(args=None):
    rclpy.init(args=args)

    print('Initialize this node only after calibration has been performed')
    input('Press <ENTER> to continue')

    # Open camera once and pass it as an argument to the vision node
    print('Opening camera')
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print('Camera opened')
    
    computerVisionServer = computerVisionNode(cap)
    rclpy.spin(computerVisionServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()