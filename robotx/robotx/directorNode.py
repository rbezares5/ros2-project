      
import sys  # get access to command line input arguments
import rclpy
from rclpy.node import Node

# import custom interfaces
from interfaces.srv import CameraCalibration
from interfaces.srv import ComputerVision  
from interfaces.srv import CheckersPlay
from interfaces.srv import RobotMovement

# define classes for the client nodes we are going to use
class CalibrationClientAsync(Node):

    def __init__(self):
        super().__init__('camera_calibration_client')    #initilize the node with this name
        self.cli = self.create_client(CameraCalibration, 'camera_calibration_service')     #type and name of the service  
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = CameraCalibration.Request()                                   

    def send_request(self):
        self.req.request = True                
        self.future = self.cli.call_async(self.req)

class VisionClientAsync(Node):

    def __init__(self):
        super().__init__('computer_vision_client')    #initilize the node with this name
        self.cli = self.create_client(ComputerVision, 'computer_vision_service')     #type and name of the service  
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ComputerVision.Request()                                   

    def send_request(self):
        self.req.request = True                
        self.future = self.cli.call_async(self.req)

class AgentClientAsync(Node):

    def __init__(self,name,board):
        super().__init__(str(name))    #initilize the node with this name
        self.cli = self.create_client(CheckersPlay, 'checkers_play_service')     #type and name of the service  
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = CheckersPlay.Request()
        self.board = board                             

    def send_request(self):
        self.req.request = True
        self.req.board = self.board             
        self.future = self.cli.call_async(self.req)

class RobotClientAsync(Node):

    def __init__(self,board):
        super().__init__('robot_client')    #initilize the node with this name
        self.cli = self.create_client(RobotMovement, 'robot_movement_service')     #type and name of the service  
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RobotMovement.Request()
        self.board = board                             

    def send_request(self):
        self.req.request = True
        self.req.board = self.board             
        self.future = self.cli.call_async(self.req)


# MAIN FUNCTION OF THE APPLICATION
def main(args=None):
    rclpy.init(args=args)

    print('Begin application.')
    input('Press <ENTER> to continue')

    print('Calibration is going to start')
    input('Press <ENTER> to continue')

    # create a client node object for the calibration service
    calibrationClient = CalibrationClientAsync()
    calibrationClient.send_request()
    
    # this loop checks if there is an available service with a matching name and type as the client
    while rclpy.ok():
        rclpy.spin_once(calibrationClient)
        if calibrationClient.future.done():
            try:
                response = calibrationClient.future.result()
            except Exception as e:
                calibrationClient.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                if response.goal == True:
                    calibrationClient.get_logger().info(
                        'Camera calibration performed successfully'                               
                        )

                    coordsList=response.coordinates
                    print(len(coordsList))
                    print(coordsList)
                break
            break
    calibrationClient.destroy_node()
    #print(len(coordsList))
    #print(coordsList)
    print('Calibration finished')  
    input('Press <ENTER> to continue')

    
    #Main Game Loop
    while True:
        # create a client node object for the computer vision service
        print('Requesting board vision')
        input('Press <ENTER> to continue')
        computerVisionClient = VisionClientAsync()
        computerVisionClient.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(computerVisionClient)
            if computerVisionClient.future.done():
                try:
                    response = computerVisionClient.future.result()
                except Exception as e:
                    computerVisionClient.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        computerVisionClient.get_logger().info(
                            'Board state analyzed successfully'                               
                            )

                        boardState=response.board
                        print(boardState)
                    break
                break
        computerVisionClient.destroy_node()

        print('Board state acquired')
        input('Press <ENTER> to continue')


        # create a client node object for the checkers play service
        print('Requesting player 1 service')
        input('Press <ENTER> to continue')
        agentPlayer1 = AgentClientAsync('player1',boardState)
        agentPlayer1.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(agentPlayer1)
            if agentPlayer1.future.done():
                try:
                    response = agentPlayer1.future.result()
                except Exception as e:
                    agentPlayer1.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        agentPlayer1.get_logger().info(
                            'Checkers play made'                               
                            )

                        boardState=response.board
                        print(boardState)
                    break
                break

        agentPlayer1.destroy_node()
        print('Player 1 makes their move')
        input('Press <ENTER> to continue')


        # create a client node object for the robot movement service
        print('Requesting robot movement service')
        input('Press <ENTER> to continue')
        robotClient = RobotClientAsync(boardState)
        robotClient.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(robotClient)
            if robotClient.future.done():
                try:
                    response = robotClient.future.result()
                except Exception as e:
                    robotClient.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        robotClient.get_logger().info(
                            'Piece(s) moved as requested'                               
                            )

                    break
                break

        robotClient.destroy_node()
        print('Robot movement(s) performed')
        input('Press <ENTER> to continue')

        print('Requesting board vision')
        input('Press <ENTER> to continue')

        # create a client node object for the computer vision service
        computerVisionClient = VisionClientAsync()
        computerVisionClient.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(computerVisionClient)
            if computerVisionClient.future.done():
                try:
                    response = computerVisionClient.future.result()
                except Exception as e:
                    computerVisionClient.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        computerVisionClient.get_logger().info(
                            'Board state analyzed successfully'                               
                            )

                        boardState=response.board
                        print(boardState)
                    break
                break

        computerVisionClient.destroy_node()
        print('Board state acquired')
        input('Press <ENTER> to continue')


        # create a client node object for the checkers play service
        print('Requesting player 2 service')
        input('Press <ENTER> to continue')
        agentPlayer2 = AgentClientAsync('player2',boardState)
        agentPlayer2.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(agentPlayer2)
            if agentPlayer2.future.done():
                try:
                    response = agentPlayer2.future.result()
                except Exception as e:
                    agentPlayer1.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        agentPlayer2.get_logger().info(
                            'Checkers play made'                               
                            )

                        boardState=response.board
                        print(boardState)
                    break
                break
        agentPlayer2.destroy_node()
        print('Player 2 makes their move')
        input('Press <ENTER> to continue')


        # create a client node object for the robot movement service
        print('Requesting robot movement service')
        input('Press <ENTER> to continue')
        robotClient = RobotClientAsync(boardState)
        robotClient.send_request()
        
        # this loop checks if there is an available service with a matching name and type as the client
        while rclpy.ok():
            rclpy.spin_once(robotClient)
            if robotClient.future.done():
                try:
                    response = robotClient.future.result()
                except Exception as e:
                    robotClient.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    if response.goal == True:
                        robotClient.get_logger().info(
                            'Piece(s) moved as requested'                               
                            )

                    break
                break
        robotClient.destroy_node()
        print('Robot movement(s) performed')
        input('Press <ENTER> to continue')


    print("That's all folks!")
    input('Press <ENTER> to end application')

    # destroy the nodes
    calibrationClient.destroy_node()
    computerVisionClient.destroy_node()
    agentPlayer1.destroy_node()
    agentPlayer2.destroy_node()
    robotClient.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()