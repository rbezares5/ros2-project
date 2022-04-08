import rclpy
from rclpy.node import Node

# import custom interfaces

from interfaces.srv import CheckersPlay


class checkersAgentNode(Node):

    def __init__(self):
        super().__init__('checkers_agent_server') #initilize the node with this name
        self.srv = self.create_service(CheckersPlay, 'checkers_play_service', self.checkersPlayCallback) #type, name and callback of the service
                
        
     

    def checkersPlayCallback(self, request, response):
        #self.get_logger().info('Incoming location request\n [x,y,z] = [%f,%f,%f]' % (request.x, request.y, request.z))
        self.get_logger().info('Checkers play request acknowledged')   #receive the request
        print('Current board state:')
        print(request.board)

        if request.request:
            # Here we get the matrix representing the board state and select a play and return the modified matrix
            boardState = request.board
            
            print('Play selected. Next board state:')
            print(boardState)
            
            response.board=boardState
            response.goal = True

        else:
            response.goal = False
        
        return response



def main(args=None):
    rclpy.init(args=args)

    checkersAgentServer = checkersAgentNode()
    rclpy.spin(checkersAgentServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()