import math
import copy
from functools import reduce
import copy
from abc import ABC, abstractmethod

class Board:
    """
    A class to represent and play an 8x8 game of checkers.
    """
    EMPTY_SPOT = 0
    P1 = 1
    P2 = 2
    P1_K = 3
    P2_K = 4
    BACKWARDS_PLAYER = P2
    HEIGHT = 8
    WIDTH = 4

    P1_SYMBOL = 'o'
    P1_K_SYMBOL = 'O'
    P2_SYMBOL = 'x'
    P2_K_SYMBOL = 'X'


    def __init__(self, old_spots=None, the_player_turn=True):
        """
        Initializes a new instance of the Board class.  Unless specified otherwise, 
        the board will be created with a start board configuration.

        the_player_turn=True indicates turn of player P1

        NOTE:
        Maybe have default parameter so board is 8x8 by default but nxn if wanted.
        """
        self.player_turn = the_player_turn
        if old_spots is None:
            self.spots = [[j, j, j, j] for j in [self.P1, self.P1, self.P1, self.EMPTY_SPOT, 
                                                self.EMPTY_SPOT, self.P2, self.P2, self.P2]]
        else:
            self.spots = old_spots


    def reset_board(self):
        """
        Resets the current configuration of the game board to the original 
        starting position.
        """
        self.spots = Board().spots


    def empty_board(self):
        """
        Removes any pieces currently on the board and leaves the board with nothing but empty spots.
        """
        # TODO Make sure [self.EMPTY_SPOT]*self.HEIGHT] has no issues
        self.spots = [[j, j, j, j] for j in [self.EMPTY_SPOT] * self.HEIGHT]   

    
    def is_game_over(self):
        """
        Finds out and returns weather the game currently being played is over or
        not.
        """
        if not self.get_possible_next_moves():
            return True

        return False


    def not_spot(self, loc):
        """
        Finds out of the spot at the given location is an actual spot on the game board.
        """
        if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or \
            loc[1] > self.WIDTH - 1:
            return True
        return False


    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        
        NOTE:
        Might want to not use this for the sake of computational time.
        """
        return self.spots[loc[0]][loc[1]]


    def forward_n_locations(self, start_loc, n, backwards=False):
        """
        Gets the locations possible for moving a piece from a given location diagonally
        forward (or backwards if wanted) a given number of times(without directional change midway).
        """
        if n % 2 == 0:
            temp1 = 0
            temp2 = 0
        elif start_loc[0] % 2 == 0:
            temp1 = 0
            temp2 = 1 
        else:
            temp1 = 1
            temp2 = 0

        answer = [[start_loc[0], start_loc[1] + math.floor(n / 2) + temp1], 
                    [start_loc[0], start_loc[1] - math.floor(n / 2) - temp2]]

        if backwards: 
            answer[0][0] = answer[0][0] - n
            answer[1][0] = answer[1][0] - n
        else:
            answer[0][0] = answer[0][0] + n
            answer[1][0] = answer[1][0] + n

        if self.not_spot(answer[0]):
            answer[0] = []
        if self.not_spot(answer[1]):
            answer[1] = []

        return answer


    def get_simple_moves(self, start_loc):
        """
        Gets the possible moves a piece can make given that it does not capture any 
        opponents pieces.

        PRE-CONDITION:
        -start_loc is a location with a players piece
        """
        if self.spots[start_loc[0]][start_loc[1]] > 2:
            next_locations = self.forward_n_locations(start_loc, 1)
            next_locations.extend(self.forward_n_locations(start_loc, 1, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next_locations = self.forward_n_locations(start_loc, 1, True)  
        else:
            next_locations = self.forward_n_locations(start_loc, 1)
        

        possible_next_locations = []

        for location in next_locations:
            if len(location) != 0:
                if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
                    possible_next_locations.append(location)
            
        return [[start_loc, end_spot] for end_spot in possible_next_locations]


    def get_capture_moves(self, start_loc, move_beginnings=None):
        """
        Recursively get all of the possible moves for a piece which involve capturing an 
        opponent's piece.
        """
        if move_beginnings is None:
            move_beginnings = [start_loc]
            
        answer = []
        if self.spots[start_loc[0]][start_loc[1]] > 2:  
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
            next1.extend(self.forward_n_locations(start_loc, 1, True))
            next2.extend(self.forward_n_locations(start_loc, 2, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next1 = self.forward_n_locations(start_loc, 1, True)
            next2 = self.forward_n_locations(start_loc, 2, True)
        else:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
        
        
        for j in range(len(next1)):
            # if both spots exist
            if (not self.not_spot(next2[j])) and (not self.not_spot(next1[j])) : 
                # if next spot is opponent
                if self.get_spot_info(next1[j]) != self.EMPTY_SPOT and \
                    self.get_spot_info(next1[j]) % 2 != self.get_spot_info(start_loc) % 2:  
                    # if next next spot is empty
                    if self.get_spot_info(next2[j]) == self.EMPTY_SPOT:
                        temp_move1 = copy.deepcopy(move_beginnings)
                        temp_move1.append(next2[j])
                        
                        answer_length = len(answer)
                        
                        if self.get_spot_info(start_loc) != self.P1 or \
                            next2[j][0] != self.HEIGHT - 1: 
                            if self.get_spot_info(start_loc) != self.P2 or next2[j][0] != 0: 

                                temp_move2 = [start_loc, next2[j]]
                                
                                temp_board = Board(copy.deepcopy(self.spots), self.player_turn)
                                temp_board.make_move(temp_move2, False)

                                answer.extend(temp_board.get_capture_moves(temp_move2[1], temp_move1))
                                
                        if len(answer) == answer_length:
                            answer.append(temp_move1)
                            
        return answer


    def get_piece_locations(self):
        """
        Gets all the pieces of the current player
        """
        piece_locations = []
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if (self.player_turn == True and 
                    (self.spots[j][i] == self.P1 or self.spots[j][i] == self.P1_K)) or \
                (self.player_turn == False and 
                    (self.spots[j][i] == self.P2 or self.spots[j][i] == self.P2_K)):
                    piece_locations.append([j, i])  

        return piece_locations        
    
        
    def get_possible_next_moves(self):
        """
        Gets the possible moves that can be made from the current board configuration.
        """

        piece_locations = self.get_piece_locations()

        try:  #Should check to make sure if this try statement is still necessary 
            capture_moves = list(reduce(lambda a, b: a + b, list(map(self.get_capture_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY

            if len(capture_moves) != 0:
                return capture_moves

            return list(reduce(lambda a, b: a + b, list(map(self.get_simple_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY
        except TypeError:
            return []
    
    def make_move(self, move, switch_player_turn=True):
        """
        Makes a given move on the board, and (as long as is wanted) switches the indicator for
        which players turn it is.
        """

        if abs(move[0][0] - move[1][0]) == 2:
            for j in range(len(move) - 1):
                if move[j][0] % 2 == 1:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j][1]
                    else:
                        middle_y = move[j + 1][1]
                else:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j + 1][1]
                    else:
                        middle_y = move[j][1]
                        
                self.spots[int((move[j][0] + move[j + 1][0]) / 2)][middle_y] = self.EMPTY_SPOT


        self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        if move[len(move) - 1][0] == self.HEIGHT - 1 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P1:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P1_K
        elif move[len(move) - 1][0] == 0 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P2:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P2_K
        else:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        self.spots[move[0][0]][move[0][1]] = self.EMPTY_SPOT

        if switch_player_turn:
            self.player_turn = not self.player_turn

    def get_potential_spots_from_moves(self, moves):
        """
        Get's the potential spots for the board if it makes any of the given moves.
        If moves is None then returns it's own current spots.
        """
        if moves is None:
            return self.spots
        answer = []
        for move in moves:
            original_spots = copy.deepcopy(self.spots)
            self.make_move(move, switch_player_turn=False)
            answer.append(self.spots) 
            self.spots = original_spots 
        
        return answer

    def insert_pieces(self, pieces_info):
        """
        Inserts a set of pieces onto a board.
        pieces_info is in the form: [[vert1, horz1, piece1], [vert2, horz2, piece2], ..., [vertn, horzn, piecen]]
        """
        for piece_info in pieces_info:
            self.spots[piece_info[0]][piece_info[1]] = piece_info[2]


    def get_symbol(self, location):
        """
        Gets the symbol for what should be at a board location.
        """
        if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
            return " "
        elif self.spots[location[0]][location[1]] == self.P1:
            return self.P1_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P2:
            return self.P2_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P1_K:
            return self.P1_K_SYMBOL
        else:
            return self.P2_K_SYMBOL


    def print_board(self):
        """
        Prints a string representation of the current game board.
        """

        index_columns = "   "
        for j in range(self.WIDTH):
            index_columns += " " + str(j) + "   " + str(j) + "  "
        print(index_columns)

        norm_line = "  |---|---|---|---|---|---|---|---|"
        print(norm_line)

        for j in range(self.HEIGHT):
            temp_line = str(j) + " "
            if j % 2 == 1:
                temp_line += "|///|"
            else:
                temp_line += "|"
            for i in range(self.WIDTH):
                temp_line = temp_line + " " + self.get_symbol([j, i]) + " |"
                if i != 3 or j % 2 != 1:  # TODO should figure out if this 3 should be changed to self.WIDTH-1
                    temp_line = temp_line + "///|"
            print(temp_line)
            print(norm_line)

class GameState:
    """
    A class which stores information about the state of a game.
    This class uses class Board to perform moves and to check whether game is won or lost.
    """


    def __init__(self, prev_state=None, the_player_turn=True):
        """
        prev_state: an instance of GameState or None
        """

        if prev_state is None:
            prev_spots = None
        else:
            prev_spots = copy.deepcopy(prev_state.board.spots)

        self.board = Board(prev_spots, the_player_turn)
        self.max_moves_done = False

    def get_num_agents(self):
        return 2

    def get_legal_actions(self):
        """
        Returns the legal moves as list of moves. A single move is a list of positions going from
        first position to next position
        """
        return self.board.get_possible_next_moves()

    def generate_successor(self, action, switch_player_turn=True):
        """
        action is a list of positions indicating move from position at first index to position at
        next index

        Returns: a new state without any changes to current state
        """

        successor_state = GameState(self, self.board.player_turn)
        successor_state.board.make_move(action, switch_player_turn)

        return successor_state

    def print_board(self):
        self.board.print_board()

    def is_first_agent_turn(self):
        """
        Returns: True if it is the turn of first agent else returns False
        """
        return self.board.player_turn
    
    def is_game_over(self):
        """
        Returns: True if either agent has won the game
        """
        return self.board.is_game_over() or self.max_moves_done

    def get_pieces_and_kings(self, player=None):
        """
        player: True if for the first player, false for the second player, None for both players

        Returns: the number of pieces and kings for every player in the current state
        """
        spots = self.board.spots

        # first agent pawns, second agent pawns, first agent kings, second agent kings
        count = [0,0,0,0]   
        for x in spots:
            for y in x:
                if y != 0:
                    count[y-1] = count[y-1] + 1

        if player is not None:
            if player:
                return [count[0], count[2]]  #Player 1
            else:
                return [count[1], count[3]]  #Player 2
        else:
            return count

class Agent(ABC):

    def __init__(self, is_learning_agent=False):
        self.is_learning_agent = is_learning_agent
        self.has_been_learning_agent = is_learning_agent

    @abstractmethod
    def get_action(self, state):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass

class AlphaBetaAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth

    def evaluation_function(self, state, agent=True):
        """
        state: the state to evaluate
        agent: True if the evaluation function is in favor of the first agent and false if
               evaluation function is in favor of second agent

        Returns: the value of evaluation
        """
        agent_ind = 0 if agent else 1
        other_ind = 1 - agent_ind

        if state.is_game_over():
            if agent and state.is_first_agent_win():
                return 500

            if not agent and state.is_second_agent_win():
                return 500

            return -500

        pieces_and_kings = state.get_pieces_and_kings()
        return pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - \
        (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2])

    def get_action(self, state):

        def mini_max(state, depth, agent, A, B):
            if agent >= state.get_num_agents():
                agent = 0

            depth += 1
            if depth == self.depth or state.is_game_over():
                return [None, self.evaluation_function(state, max_agent)]
            elif agent == 0:
                return maximum(state, depth, agent, A, B)
            else:
                return minimum(state, depth, agent, A, B)

        def maximum(state, depth, agent, A, B):
            output = [None, -float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent + 1, A, B)

                check = val[1]

                if check > output[1]:
                    output = [action, check]

                if check > B:
                    return [action, check]

                A = max(A, check)

            return output

        def minimum(state, depth, agent, A, B):
            output = [None, float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent+1, A, B)

                check = val[1]

                if check < output[1]:
                    output = [action, check]

                if check < A:
                    return [action, check]

                B = min(B, check)

            return output

        # max_agent is true meaning it is the turn of first player at the state in 
        # which to choose the action
        max_agent = state.is_first_agent_turn()
        output = mini_max(state, -1, 0, -float("inf"), float("inf"))
        return output[0]
