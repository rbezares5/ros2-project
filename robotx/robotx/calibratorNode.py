import rclpy
from rclpy.node import Node

import numpy as np
import cv2 as cv
import copy

# import custom interfaces
from interfaces.srv import CameraCalibration


class cameraCalibratorNode(Node):

    def __init__(self):
        super().__init__('camera_calibration_server') #initilize the node with this name
        self.srv = self.create_service(CameraCalibration, 'camera_calibration_service', self.cameraCalibrationCallback) #type, name and callback of the service        

    def cameraCalibrationCallback(self, request, response):
        self.get_logger().info('Calibration request acknowledged')   #receive the request

        if request.request:
            #Here we get the image and start working on it
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera")
                exit()

            #firstly, we are going to just get a roi of the board
            _, frame = cap.read()
            r=cv.selectROI(frame)

            #convert the roi, which is a tuple, into a list so that it can be passed in the response
            roi=list(r)
            response.roi=roi

            print('Detecting empty board. Press q to continue')
            while True:
                _, frame = cap.read()
                frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                #Detect chessboard squares using canny
                grayBlur = cv.GaussianBlur(gray, (15, 15), 2)
                edges = cv.Canny(grayBlur,40,120)
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
                edges = cv.dilate(edges, kernel)
                edges = cv.bitwise_not(edges)

                # find contours
                contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

                # filter out contours by size
                squares = []
                for con in contours:
                    area = cv.contourArea(con)
                    if (area < 1000) & (area > 200): # size threshold
                        squares.append(con)

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
                    i+=1

                # We have to order the centroids of the squares
                if len(centroidsSquares) > 0:
                    centroidsSquares=np.asarray(centroidsSquares)
                    # bin the "y" coordinate of the centroids
                    _, bins = np.histogram(centroidsSquares[:,1], bins='auto')
                    bins=bins-5 #we shift the bins edges to account for approximation error
                    centroidsSquaresOrder=np.digitize(centroidsSquares[:,1],bins,right=False)
                    centroidsSquares2=copy.deepcopy(centroidsSquares)
                    # assign the bin order value over the actual coordinate
                    centroidsSquares2[:,1]=centroidsSquaresOrder
                    # reorder the points based on the y bins and then on the x coordinate
                    index=np.lexsort((centroidsSquares2[:,0],centroidsSquares2[:,1]))
                    centroidsSquares3=[(centroidsSquares[i,0],centroidsSquares[i,1]) for i in index]
                    centroidsSquares3=np.asarray(centroidsSquares3)
               
                    i=1
                    for j in range(len(centroidsSquares3)):
                        cX=centroidsSquares3[j,0]
                        cY=centroidsSquares3[j,1]
                        cv.putText(frame, str(i), (cX, cY),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        i+=1
                    cv.imshow('Squares centroids', frame)

                # convert the array into a list so that it can be passed
                #coordsList=centroidsSquares3.flatten()
                #coordsList=coordsList.tolist()


                if len(centroidsSquares3)==64:
                    print('Full board detected. Reducing to playable squares only')
                    # turn CentroidsSquares3 into a reduced array of 32 coordinates, discarding non-playable squares
                    centroidsSquares4=np.zeros((32,2), dtype=int)
                    k=0
                    for i in range(8):
                        for j in range(8):
                            if (i+j+1)%2==1:
                                centroidsSquares4[k]=centroidsSquares3[np.ravel_multi_index((i,j),(8,8))]
                                k+=1

                    # print the reduced coordinates onscreen
                    i=1
                    for j in range(len(centroidsSquares4)):
                        cX=centroidsSquares4[j,0]
                        cY=centroidsSquares4[j,1]
                        cv.putText(frame, str(i), (cX, cY),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        i+=1
                    cv.imshow('Squares centroids', frame)

                    # convert the array into a list so that it can be passed
                    coordsList=centroidsSquares4.flatten()
                    coordsList=coordsList.tolist()

                    break

                if cv.waitKey(1) == ord('q'):
                    break

            response.goal = True
            response.coordinates = coordsList

        else:
            response.goal = False
        
        return response


def main(args=None):
    rclpy.init(args=args)

    calibratorServer = cameraCalibratorNode()

    rclpy.spin(calibratorServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()