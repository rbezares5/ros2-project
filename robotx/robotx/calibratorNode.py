from interfaces.srv import CameraCalibration

import rclpy
from rclpy.node import Node


class cameraCalibratorNode(Node):

    def __init__(self):
        super().__init__('camera_calibration_server') #initilize the node with this name
        self.srv = self.create_service(CameraCalibration, 'camera_calibration_service', self.cameraCalibrationCallback) #type, name and callback of the service        

    def cameraCalibrationCallback(self, request, response):
        #self.get_logger().info('Incoming location request\n [x,y,z] = [%f,%f,%f]' % (request.x, request.y, request.z))
        self.get_logger().info('Calibration request acknowledged')   #receive the request

        if request.request:
             
            response.goal = True

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