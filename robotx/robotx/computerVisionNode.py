import rclpy
from rclpy.node import Node

import numpy as np
import cv2 as cv
import copy

# import custom interfaces
from interfaces.srv import ComputerVision

# define color segmentation functions
def createMaskR(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0
    hMax=0.195
    sMin=0.410
    sMax=1
    vMin=0.646
    vMax=1
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask

def createMaskB(frame):
    # HSV filter
    hsv= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #hsv thresholds
    hMin=0.484
    hMax=0.702
    sMin=0.451
    sMax=1
    vMin=0.421
    vMax=0.846
    lower=np.array([180*hMin, 255*sMin, 255*vMin], np.uint8)
    upper=np.array([180*hMax, 255*sMax, 255*vMax], np.uint8)

    mask=cv.inRange(hsv, lower, upper)
    
    return mask   


class computerVisionNode(Node):

    def __init__(self):
        super().__init__('computer_vision_server') #initilize the node with this name
        self.srv = self.create_service(ComputerVision, 'computer_vision_service', self.computerVisionCallback) #type, name and callback of the service

    def computerVisionCallback(self, request, response):
        self.get_logger().info('Vision request acknowledged')   #receive the request

        if request.request:
            # Here we process the image and finally create a matrix representing the board state

            # COMPUTER VISION CODE
            # create camera object
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()

            # Capture frame-by-frame
            _, frame = cap.read()

            # Our operations on the frame come here
            #crop the image using the ROI
            r=request.roi
            frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            blur = cv.GaussianBlur(frame, (15, 15), 2)

            #color segmentation
            maskR=createMaskR(blur)
            maskB=createMaskB(blur)

            #morphological operations to get only the pieces
            kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))

            #red pieces
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

            #Now we compare the coordinates of each piece with the coordinates of each square
            boardState = np.zeros((8,8), dtype=int)
            centroidsSquares = request.boardcoords
            centroidsSquares = np.reshape(centroidsSquares, (64,2))

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
                    boardState[np.unravel_index(pos,(8,8))]=1 #linear index

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
                    boardState[np.unravel_index(pos,(8,8))]=2 #linear index

            # convert the array into a list so that it can be passed
            boardState=boardState.flatten() 
            boardState=boardState.tolist()

            response.board=boardState
            response.goal = True

        else:
            response.goal = False
        
        return response


def main(args=None):
    rclpy.init(args=args)

    computerVisionServer = computerVisionNode()
    rclpy.spin(computerVisionServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()