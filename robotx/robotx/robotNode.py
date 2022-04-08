import rclpy
from rclpy.node import Node

# import custom interfaces

from interfaces.srv import RobotMovement


class robotServerNode(Node):

    def __init__(self):
        super().__init__('robot_server') #initilize the node with this name
        self.srv = self.create_service(RobotMovement, 'robot_movement_service', self.robotCallback) #type, name and callback of the service
                 

    def robotCallback(self, request, response):
        #self.get_logger().info('Incoming location request\n [x,y,z] = [%f,%f,%f]' % (request.x, request.y, request.z))
        self.get_logger().info('Robot movement request acknowledged')   #receive the request
        #print('Current board state:')
        #print(request.board)

        if request.request:
            # Here we get the matrix representing the new board state, we have to somehow detect which pieces to move
                                                                    #maybe that should be done by the player node??
            boardState = request.board
            boardState
            
            print('Moving piece(s)...')

            print('Piece(s) moved!')
            
            response.goal = True

        else:
            response.goal = False
        
        return response



def main(args=None):
    rclpy.init(args=args)

    robotServer = robotServerNode()
    rclpy.spin(robotServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()