import numpy as np

from checkersClasses import *


#Generate a board in numpy array form (this should be generated in the vision node)
boardNumpy=np.zeros((8,8), dtype=int)
for i in range(3):
    for j in range(8):
        if (i+j+1)%2==1:
            boardNumpy[i,j]=1

for i in range(5,8):
    for j in range(8):
        if (i+j+1)%2==1:
            boardNumpy[i,j]=2

print("Board state as numpy array")
print(boardNumpy)

'''
#From the array, get only the playable squares and turn it into a list usable by the checkers program
boardList=np.zeros((8,4), dtype=int)
k=0
for i in range(8):
    for j in range(8):
        if (i+j+1)%2==1:
            boardList[np.unravel_index(k,(8,4))]=boardNumpy[i,j]
            k+=1

boardList=boardList.tolist()
#print(boardList)
'''

# This is the actual data we get from the computer vision node
boardRequest=boardNumpy.flatten()
boardRequest=boardRequest.tolist()

print("Board state as list from the request data")
print(boardRequest)

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
print("Board state as a list using the same format as the game program")
print(boardList)

#Initialize a board object and asign the values of the list we just generated

#myBoard=Board()
#print(myBoard.spots)

myBoard=GameState()


myBoard.board.spots=boardList
print(myBoard.board.spots)
print("ingame board")
myBoard.print_board()

aiPlayer = AlphaBetaAgent(depth=3)
move=aiPlayer.get_action(myBoard)
print(move)

myBoard.board.make_move(move, switch_player_turn=False)
#myBoard.make_move(move, switch_player_turn=False)
#print(myBoard.spots)
myBoard.print_board()

'''
move=aiPlayer.get_action(myBoard)
print(move)

myBoard.board.make_move(move, switch_player_turn=False)
myBoard.print_board()
'''

'''
while True:
    move=aiPlayer.get_action(myBoard)
    print(move)

    myBoard.board.make_move(move)
    myBoard.print_board()
'''