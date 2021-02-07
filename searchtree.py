"""
SearchTree Class for Chess Bot
Jerry Shukai Zhang
12/06/2020 - 02/03/2021
"""

from node import Node
import numpy as np


# Constants:
MAX_LEVEL = 4  # Max level of Minimax algorithm


class SearchTree:
    def __init__(self, player):
        self.player = player  # the side that computer will play as

        # Layout of the starting board
        # Contains entries of format [Side, ID, Piece]
        # Empty squares are [-1, -1, -1]
        start_board = np.array([[[-1, -1, -1] for col in range(8)] for row in\
            range(8)])

        # Set my_moved and opp_moved for the starting board
        white_squares = np.array([[-1, -1] for _ in range(16)])
        black_squares = np.array([[-1, -1] for _ in range(16)])

        # Fill in the starting board
        for side in range(2):  # 0 = White; 1 = Black
            # Fill in the appropriate squares
            squares = white_squares if side == 0 else black_squares

            for row in range(2):  # Each side starts in 2 rows
                for col in range(8):  # With 8 pieces in each
                    # Determine coordinates of the piece
                    x = col
                    y = row if side == 0 else 7 if row == 0 else 6

                    # Determine the piece
                    if row == 1:
                        piece = 5  # Pawn
                    elif col == 0 or col == 7:
                        piece = 0  # Rook
                    elif col == 1 or col == 6:
                        piece = 1  # Knight
                    elif col == 2 or col == 5:
                        piece = 2  # Bishop
                    elif col == 3:
                        piece = 3  # Queen
                    else:
                        piece = 4  # King

                    # Calculate the ID
                    piece_id = row * 8 + col

                    # Set squares fields
                    squares[piece_id] = np.array([y, x])

                    # Set the piece
                    start_board[y, x] = np.array([side, piece_id, piece])

        # Fill in the values for the starting board
        self.curr_board = Node(start_board, 0, player)  # White goes first

        self.curr_board.my_squares = white_squares
        self.curr_board.opp_squares = black_squares

        self.curr_board.my_moved = np.array([False for _ in range(16)])
        self.curr_board.opp_moved = np.array([False for _ in range(16)])

        # Fill in the targeted matrices
        white_targeted = np.array([[0 for _ in range(8)]\
            for _ in range(8)])
        black_targeted = np.array([[0 for _ in range(8)]\
            for _ in range(8)])

        first_row = np.array([0, 7])
        second_row = np.array([1, 6])
        third_row = np.array([2, 5])

        # Starting board will always have the same targeting schemes
        for side in range(2):
            targeted = white_targeted if side == 0 else black_targeted

            row_1 = first_row[side]
            row_2 = second_row[side]
            row_3 = third_row[side]

            for col in range(8):
                num_1 = 0 if col == 0 or col == 7 else 1
                num_2 = 3 if col == 3 or col == 4 else 1
                num_3 = 3 if col == 2 or col == 5 else 2

                targeted[row_1, col] = num_1
                targeted[row_2, col] = num_2
                targeted[row_3, col] = num_3

        self.curr_board.my_targeted = white_targeted
        self.curr_board.opp_targeted = black_targeted

        self.moves_made = []  # Will list all the moves made by the computer
        self.local_nodes_generated = 0  # Nodes generated in one search
        self.total_nodes_generated = 0  # Total nodes generated

    # Minimax algorithm with Alpha-Beta pruning; tree-like
    # Only goes up to max_level before stopping
    def find_next_move(self):
        self.local_nodes_generated = 0  # Number of nodes generated in this search

        # Returns val, move
        # check_node = node that is being expanded/inspected
        # alpha = highest heuristic value so far
        # beta = lowest heuristic value so far
        # level = how many lookaheads have been done so far
        def max_value(check_node, alpha, beta, level):
            # Call expand() first because this updates Checkmate/Draw status
            child_nodes = check_node.expand()

            # If game is over or node reaches cutoff
            if check_node.outcome == "Checkmate" or check_node.outcome\
                    == "Draw" or level == MAX_LEVEL:
                return check_node.h_value, None

            val = -9999

            for node in child_nodes:
                temp_val, temp_move = min_value(node, alpha, beta, level + 1)

                # Update val/move if a node with a higher value is found
                # Also update alpha accordingly
                if temp_val > val:
                    val, move = temp_val, node.move
                    alpha = max(alpha, val)

                # However, if value is greater than beta, this node will not
                # be reached because the opponent is assumed is assumed to
                # make the optimal move
                if val >= beta:
                    return val, move

                self.local_nodes_generated += 1
                self.total_nodes_generated += 1
            return val, move

        # Return val, move
        # check_node = node that is being expanded/inspected
        # alpha = highest heuristic value so far
        # beta = lowest heuristic value so far
        # level = how many lookaheads have been done so far
        def min_value(check_node, alpha, beta, level):
            child_nodes = check_node.expand()

            # If game is over or node reaches cutoff
            if check_node.outcome == "Checkmate" or check_node.outcome\
                    == "Draw" or level == MAX_LEVEL:
                return check_node.h_value, None

            val = 9999

            for node in child_nodes:
                temp_val, temp_move = max_value(node, alpha, beta, level + 1)

                # Update val/move if a node with a lower value is found
                # Also update beta accordingly
                if temp_val < val:
                    val, move = temp_val, temp_move
                    beta = min(beta, val)

                # However, if value is less than alpha, this node will not
                # be reached because the player is assumed is assumed to
                # make the optimal move
                if val <= alpha:
                    return val, move

                self.local_nodes_generated += 1
                self.total_nodes_generated += 1
            return val, move

        _, move = max_value(self.curr_board, -9999, 9999, 0)

        print("Nodes Generated for this move: ", self.local_nodes_generated)
        print("Total Nodes Generated: ", self.total_nodes_generated)

        return move

    # Conducts the move specified
    # move = [[old_row, old_col], [new_row, new_col]]
    def tree_do_move(self, move):
        if self.curr_board.side == self.player:
            self.moves_made.append(move)

        self.curr_board = self.curr_board.node_do_move(move)