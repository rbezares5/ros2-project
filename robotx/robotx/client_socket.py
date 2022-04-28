import socket
import numpy as np
from checkersClasses import *

# Generate numpy board
boardNumpy=np.array([
    (0,0,0,0,0,0,0,0),
    (0,0,0,0,0,0,0,0),
    (1,0,0,0,1,0,1,0),
    (0,2,0,1,0,0,0,0),
    (0,0,0,0,2,0,0,0),
    (0,0,0,2,0,0,0,0),
    (0,0,0,0,0,0,2,0),
    (0,0,0,0,0,0,0,0)
])

# This is the actual data we get from the computer vision node
boardRequest=boardNumpy.flatten()
boardRequest=boardRequest.tolist()

# Convert that list into a format that can be used by the checkers program
boardList=np.zeros((8,4), dtype=int)
k=0
for i in range(8):
    for j in range(8):
        if (i+j+1)%2==1:
            boardList[np.unravel_index(k,(8,4))]=boardRequest[np.ravel_multi_index((i,j),(8,8))]
            #boardList[np.unravel_index(k,(8,4))]=boardRequest[i+i*j]
            k+=1

boardList=boardList.tolist()

#Initialize a board object and asign the values of the list we just generated
myBoard=GameState()

myBoard.board.spots=boardList
print(myBoard.board.spots)
print("ingame board")
myBoard.print_board()

aiPlayer = AlphaBetaAgent(depth=3)
move=aiPlayer.get_action(myBoard)
print('generated move')
print(move)

cap, prom, _ = myBoard.board.make_move2(move, switch_player_turn=False)
if prom==[]:
    prom=[[0,0]]#think of a better 'error' position
if cap==[]:
    cap=[[0,0]]#think of a better 'error' position
myBoard.print_board()

# Create a TCP/IP socket
sockMoveOrigin = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sockMoveDestination = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sockPromote1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sockCapture = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Connect the socket to the port where the server is listening
address='localhost'
server_address1 = (address, 1010)
server_address2 = (address, 1020)
server_address3 = (address, 1030)
server_address4 = (address, 1040)
print('connecting to {} port {}'.format(*server_address1))
sockMoveOrigin.connect(server_address1)
sockMoveDestination.connect(server_address2)
sockPromote1.connect(server_address3)
sockCapture.connect(server_address4)

#construct message
#message = [1,3]
#message=bytearray(message)
#cant receive properly bigger than size 2 arrays???
origin=bytearray([move[0][0],move[0][1]])
destination=bytearray([move[len(move)-1][0],move[len(move)-1][1]])
promote1=bytearray([prom[0][0],prom[0][1]])
if len(cap)==2:
    capture=bytearray([cap[0][0],cap[0][1],cap[1][0],cap[1][1]])
else:
    capture=bytearray([cap[0][0],cap[0][1],0,0])
    #capture=bytearray([cap[0][0],cap[0][1]])
#print(message)

#send message
sockMoveOrigin.sendall(origin)
sockMoveDestination.sendall(destination)
sockPromote1.sendall(promote1)
sockCapture.sendall(capture)

print('closing socket')
sockMoveOrigin.close()
sockMoveDestination.close()
sockPromote1.close()
sockCapture.close()