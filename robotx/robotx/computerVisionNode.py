import rclpy
from rclpy.node import Node

# import custom interfaces
from interfaces.srv import ComputerVision


class computerVisionNode(Node):

    def __init__(self):
        super().__init__('computer_vision_server') #initilize the node with this name
        self.srv = self.create_service(ComputerVision, 'computer_vision_service', self.computerVisionCallback) #type, name and callback of the service

    def computerVisionCallback(self, request, response):
        self.get_logger().info('Vision request acknowledged')   #receive the request

        if request.request:
            # Here we process the image and finally create a matrix representing the board state
            boardState = list(range(64))
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