This is a work in progress project.

The aim is to create a ROS 2 system composed of several nodes.

The main program creates client nodes that call their respective servers. It can be started with the following command:
ros2 run robotx application

But before that the servers must be initialized individually:
ros2 run robotx calibration
ros2 run robotx vision
ros2 run robotx agent
ros2 run robotx robot

As of now, only the structure of communication between nodes is implemented.

Functionality of different nodes currently being added

Credits for the checkers game code which I adapted for this project to: https://github.com/VarunRaval48/checkers-AI