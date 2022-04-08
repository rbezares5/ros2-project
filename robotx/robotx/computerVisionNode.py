import rclpy
from rclpy.node import Node

# import custom interfaces
#from interfaces.msg import BoardState
from interfaces.srv import ComputerVision


class computerVisionNode(Node):

    def __init__(self):
        super().__init__('computer_vision_server') #initilize the node with this name
        self.srv = self.create_service(ComputerVision, 'computer_vision_service', self.computerVisionCallback) #type, name and callback of the service
                
        #self.publisher = self.create_publisher(BoardState, 'board_state', 10) #this node will also be a publisher

     

    def computerVisionCallback(self, request, response):
        #self.get_logger().info('Incoming location request\n [x,y,z] = [%f,%f,%f]' % (request.x, request.y, request.z))
        self.get_logger().info('Vision request acknowledged')   #receive the request

        if request.request:
            # Here we process the image and finally create a matrix representing the board state
            boardState = list(range(64))
            #self.board_publish(boardState)
            #response.board=boardState
            response.goal = True

        else:
            response.goal = False
        
        return response

    '''
    def board_publish(self,pBoardData):
        msg = BoardState()                 

        msg.board[:] = pBoardData 

        self.publisher.publish(msg)
        print(msg.board)
    '''   


def main(args=None):
    rclpy.init(args=args)

    computerVisionServer = computerVisionNode()
    rclpy.spin(computerVisionServer)

    rclpy.shutdown()


if __name__ == '__main__':
    main()