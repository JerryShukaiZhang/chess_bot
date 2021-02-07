"""
Node Class for Chess Bot
Jerry Shukai Zhang
12/06/2020 - 02/03/2021
"""

import numpy as np
from copy import deepcopy

POINTS = np.array([5, 3, 3, 9, 0, 1])  # Points that each piece are worth
EMPTY_2 = np.array([-1, -1])  # "Empty" two-element array used for comparison
EMPTY_3 = np.array([-1, -1, -1])  # Same idea, but three-element

# Values used in heuristics
HEUR_P1_ADD = 100  # If someone checkmates, add by this number
HEUR_P1_DIVIDE = 2  # If someone draws, divide by this number
HEUR_P4_POINTS = [3, 2, 2, 4, 0, 1]  # Points for moving each piece type from
# its starting position
HEUR_P5_ADD = 3  # Points for checking
HEUR_P6_ADD = 3  # Points for castling


class Node:
    def __init__(self, state, side, player):
        # Parameter inputs
        self.board = state  # The current board-state
        self.side = side  # Whoever has the move right now
        self.player = player  # Side that the computer is playing as

        # Values given by parent
        # My own values (values of the person making the move right now)
        self.my_squares = None  # An array containing the coordinates of every
        # piece (taken pieces have the coordinates [-1, -1])
        self.my_moved = None  # An array containing booleans indicating
        # whether or not every piece has moved already (taken pieces are set
        # to False)
        self.my_targeted = None  # A matrix containing ints indicating how
        # many times this player's pieces are attacking every square
        self.en_passant = np.array([-1, -1])  # Holds en passant coordinate

        # Opponent's values
        self.opp_squares = None
        self.opp_moved = None
        self.opp_targeted = None

        self.h_value = 0  # Heuristic value
        self.outcome = "Ongoing"  # "Ongoing", "Draw", "Checkmate"
        self.move = None  # Most recent move used to create this board state

    # Returns the number of times that pieces in "squares" are attacked in
    # the "targeted" matrix
    # squares = the pieces that are being attacked
    # targeted = the matrix that indicates the number of times that the pieces
    # are being attacked
    def num_attacked(self, squares, targeted):
        total = 0

        for square in squares:
            if (square != EMPTY_2).all():
                total += targeted[square[0], square[1]]

        return total

    # Used when a square is cleared or blocked (moved/placed in empty square)
    # side = the side of the targeted matrix
    # square = the square being altered
    # targeted = the matrix to be changed
    # board = the board state after the square is altered
    # action = 0 if the square is being cleared, 1 if it's being blocked
    def alter_targeted(self, side, square, targeted, board, action):
        # Coordinates of the square to check
        start_row = square[0]
        start_col = square[1]

        # +1 to each square if a square is being cleared, -1 if it's blocked
        change = 1 if action == 0 else -1

        # Check the cardinal directions for Rook/Queen
        # Find the coordinates of the upmost/rightmost/downmost/leftmost
        # pieces (-1 set the default for min values and 8 for max)
        vert = np.array([-1, 8])
        horiz = np.array([-1, 8])
        for direction in range(4):
            start_index = start_row if direction < 2 else start_col
            start_index += -1 if direction % 2 == 0 else 1
            end_index = vert[direction] if direction < 2 else\
                horiz[direction % 2]
            increment = -1 if direction % 2 == 0 else 1

            for index in range(start_index, end_index, increment):
                row = index if direction < 2 else start_row
                col = start_row if direction < 2 else index

                if (board[row, col] != EMPTY_3).all():
                    if direction < 2:
                        vert[direction] = index
                    else:
                        horiz[direction % 2] = index

                    break

        # Once the needed pieces of found, determine how many times each
        # square in this row/col need to have their numbers increased
        for direction in range(4):
            index = vert[direction] if direction < 2 else horiz[direction % 2]
            increment = 1 if direction % 2 == 0 else -1

            if index != -1 and index != 8:
                row = index if direction < 2 else start_row
                col = start_col if direction < 2 else index

                tup = board[row, col]
                piece = tup[2]

                if tup[0] == side and (piece == 0 or piece == 3):
                    while row in range(0, 8) and col in range(0, 8):
                        targeted[row, col] += change

                        if (board[row, col] != EMPTY_3).all():
                            break

                        if direction < 2:
                            row += increment
                        else:
                            col += increment

        # Check the intermediate directions for Bishop/Queen
        # Find the coordinates of the pieces on every diagonal
        # (-1 set the default for min values and 8 for max)
        pos_diag = np.array([-1, 8])
        neg_diag = np.array([-1, 8])

        row_change = -1
        col_change = -1
        for direction in range(4):
            row = start_row + row_change
            col = start_col + col_change

            while row in range(0, 8) and col in range(0, 8):
                if (board[row, col] != EMPTY_3).all():
                    if direction % 2 == 0:
                        pos_diag[direction // 2] = index
                    else:
                        neg_diag[(direction - 1) // 2] = index

                    break

                row += row_change
                col += col_change

        # Once the needed pieces of found, determine how many times each
        # square in each diagonal need to have their numbers increased
        for direction in range(4):
            index = vert[direction // 2] if direction % 2 == 0 else\
                horiz[(direction - 1) % 2]
            row_change = 1 if direction < 2 else -1
            col_change = 1 if direction % 2 == 0 else 1

            if index != -1 and index != 8:
                row = index
                col = index if direction % 2 == 0 else 7 - index

                tup = board[row, col]
                piece = tup[2]

                if tup[0] == side and (piece == 2 or piece == 3):
                    while row in range(0, 8) and col in range(0, 8):
                        targeted[row, col] += change

                        if (board[row, col] != EMPTY_3).all():
                            break

                        row += row_change
                        col += col_change

    # Used when a piece is inserted or removed (placed/ taken)
    # side = the side of the targeted matrix
    # square = the piece being updated
    # targeted = the matrix to be changed
    # board = the board state after the piece is updated
    # action = 0 if the square is being inserted, 1 if it's being removed
    def update_targeted(self, square, targeted, board, action):
        # Coordinates of the square to check
        start_row = square[0]
        start_col = square[1]

        # +1 to each square if a piece is being inserted, -1 if it's removed
        change = 1 if action == 0 else -1

        tup = board[start_row, start_col]
        side = tup[0]
        piece = tup[2]

        # If this piece is a Rook or a Queen
        if piece == 0 or piece == 3:
            row_change = 1
            col_change = 0

            # Alter every square in the same row/col until another piece is
            # found
            for _ in range(4):
                row = start_row + row_change
                col = start_col + col_change

                while row in range(0, 8) and col in range(0, 8):
                    targeted[row, col] += change

                    if (self.board[row, col] != EMPTY_3).all():
                        break

                    row += row_change
                    col += col_change

                row_change, col_change = col_change, row_change
                col_change *= -1

        # If the piece is a Knight
        elif piece == 1:
            row_change = 1
            col_change = 2

            # Alter each "L"
            for _ in range(8):
                row = start_row + row_change
                col = start_col + col_change

                if row in range(0, 8) and col in range(0, 8):
                    targeted[row, col] += change

                row_change, col_change = col_change, row_change
                col_change *= -1

                if _ == 3:
                    row_change *= -1
        # If this piece is a King
        elif piece == 4:
            # Check all 1-square moves
            row_change = 1
            col_change = 0

            for _ in range(8):
                row = start_row + row_change
                col = start_col + col_change

                if row in range(0, 8) and col in range(0, 8):
                    targeted[row, col] += change

                row_change, col_change = col_change, row_change
                col_change *= -1

                if _ == 3:
                    row_change = 1
                    col_change = 1
        # If this piece is a Pawn
        elif piece == 5:
            one_up = 1 if side == 0 else -1
            condition = np.array([start_col > 0, start_col < 7])
            col = np.array([start_col - 1, start_col + 1])

            # Alter the left/right diagonals
            for diag in range(2):
                if condition[diag]:
                    targeted[start_row + one_up, col[diag]] += change

        # If the piece is a Bishop or a Queen
        if piece == 2 or piece == 3:
            row_change = 1
            col_change = 1

            # Alter the square in each diagonal until another piece is found
            for _ in range(4):
                row = start_row + row_change
                col = start_col + col_change

                while row in range(0, 8) and col in range(0, 8):
                    targeted[row, col] += change

                    if (self.board[row, col] != EMPTY_3).all():
                        break

                    row += row_change
                    col += col_change

                row_change, col_change = col_change, row_change
                col_change *= -1

    # Returns True if the square is checked in this position, False otherwise
    # square = the square to check
    # side = the side that might be attacked
    def is_attacked(self, square, side):
        targeted = self.opp_targeted if side == self.side\
            else self.my_targeted

        return targeted[square[0], square[1]] > 0

    # Returns True if the node represents a valid move (the King of the
    # player who moved last is not checked)
    # side = the side whose "checked" status is in question
    def is_checked(self, side):
        # Check to make sure the opponent's king is not checked
        king_coord = self.my_squares[4] if side == self.side\
            else self.opp_squares[4]

        return self.is_attacked(king_coord, side)

    # Returns an array of available nodes/ only add valid nodes
    # Also updates game status if checkmate/draw
    def expand(self):
        opp = 1 if self.side == 0 else 0
        available_nodes = []

        for piece_id in range(len(self.my_squares)):
            # Do not check pieces that are already taken
            if (self.my_squares[piece_id] == EMPTY_2).all():
                continue

            # Coordinates of the piece being examined
            start_row = self.my_squares[piece_id][0]
            start_col = self.my_squares[piece_id][1]
            start = np.array([start_row, start_col])

            tup = self.board[start_row, start_col]
            piece = tup[2]

            # If the piece is a Rook or a Queen
            if piece == 0 or piece == 3:
                row_change = 1
                col_change = 0

                # Moves valid WHILE coordinates in range + (square empty or
                # (square = opp and square != King))
                for _ in range(4):
                    row = start_row + row_change
                    col = start_col + col_change

                    # Account for all available moves
                    while row in range(0, 8) and col in range(0, 8):
                        end = np.array([row, col])
                        move = np.array([start, end])

                        # If the square is empty
                        if (self.board[row, col] == EMPTY_3).all():
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                        # If the square is occupied
                        else:
                            # If the square is takeable
                            if self.board[row, col][0] != self.side and\
                                    self.board[row, col][2] != 4:
                                node = self.node_do_move(move)

                                if not node.is_checked(opp):
                                    available_nodes.append(node)

                            break

                        row += row_change
                        col += col_change

                    row_change, col_change = col_change, row_change
                    col_change *= -1

            # If the piece is a Knight
            elif piece == 1:
                row_change = 1
                col_change = 2

                # Knight move valid IF coordinates in range + (square empty or
                # (square = opp and square != King))
                for _ in range(8):
                    row = start_row + row_change
                    col = start_col + col_change

                    if row in range(0, 8) and col in range(0, 8):
                        end = np.array([row, col])
                        move = np.array([start, end])

                        # If the square is empty
                        if (self.board[row, col] == EMPTY_3).all():
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                        # If the square is occupied and takeable
                        elif self.board[row, col][0] != self.side and \
                                self.board[row, col][2] != 4:
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                    row_change, col_change = col_change, row_change
                    col_change *= -1

                    if _ == 3:
                        row_change *= -1

            # If the piece is a King
            elif piece == 4:
                # Check all 1-square moves
                row_change = 1
                col_change = 0

                for _ in range(8):
                    row = start_row + row_change
                    col = start_col + col_change

                    if row in range(0, 8) and col in range(0, 8):
                        end = np.array([row, col])
                        move = np.array([start, end])

                        # If the square is empty
                        if (self.board[row, col] == EMPTY_3).all():
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                        # If the square is occupied and takeable
                        elif self.board[row, col][0] != self.side and\
                                self.board[row, col][2] != 4:
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                    row_change, col_change = col_change, row_change
                    col_change *= -1

                    if _ == 3:
                        row_change = 1
                        col_change = 1

                # Represent castling
                if not self.my_moved[4] and not self.is_checked(self.side):
                    clear = np.array([True, True])  # Whether a castle on
                    # the left/right side is available
                    condition = np.array([not self.my_moved[0],\
                        not self.my_moved[7]])  # The initial condition for
                        # castle-ability on either side
                    distance = np.array([-2, 2])  # The space that the King
                    # moves for a castle on the left/right side
                    start_index = np.array([2, 5])  # The column to start
                    # looking at when looking for a castle on the left/right
                    end_index = np.array([4, 7]) # The column to stop looking
                    # at when looking for a castle on the left/right

                    for rook in range(2):
                        if condition[rook]:
                            for check in range(start_index[rook],\
                                    end_index[rook]):
                                # Just do one extra check because there is one
                                # square on a left castle where one of the
                                # squares between the Rook and King is allowed
                                # to be checked
                                if rook == 0 and (self.board[start_row, 1] !=\
                                        EMPTY_3).all():
                                    clear[rook] = False

                                    break

                                # If there is a piece in the way, or if a
                                # square in between is checked, the castle
                                # is not valid
                                if (self.board[start_row, check] !=\
                                        EMPTY_3).all() or self.is_attacked\
                                        ([start_row, check], self.side):
                                    clear[rook] = False

                                    break

                        # If the castle is valid, add it to available nodes
                        if clear[rook]:
                            end = np.array([start_row, start_col +\
                                distance[rook]])
                            move = np.array([start, end])

                            node = self.node_do_move(move)

                            available_nodes.append(node)

            # If the piece is a Pawn
            elif piece == 5:
                # On White, moving "up" increases row by 1, on Black, row
                # decreases by 1
                one_up = 1 if self.side == 0 else -1

                # The initial row for pawns on White side is 1; 6 on Black
                initial_row = 1 if self.side == 0 else 6

                # A pawn can move up if no piece is in front of it
                if (self.board[start_row + one_up, start_col] == EMPTY_3)\
                        .all():
                    end = np.array([start_row + one_up, start_col])
                    move = np.array([start, end])

                    # If a pawn is at either end of the board, do a Pawn
                    # promotion
                    if start_row + one_up == 7 or start_row + one_up == 0:
                        available_nodes += self.pawn_promotion(move)
                    else:
                        node = self.node_do_move(move)

                        if not node.is_checked(opp):
                            available_nodes.append(node)

                    # If a pawn is in its starting row, a two-up move is
                    # available as well
                    if start_row == initial_row and (self.board[start_row\
                            + (2 * one_up), start_col] == EMPTY_3).all():
                        end = np.array([start_row + (2 * one_up), start_col])
                        move = np.array([start, end])
                        
                        node = self.node_do_move(move)

                        if not node.is_checked(opp):
                            available_nodes.append(node)

                # A pawn can take on either diagonal
                condition = np.array([start_col > 0, start_col < 7])
                col_change = np.array([-1, 1])

                # If the pawn can take a piece on left, right diagonal
                for diag in range(2):
                    if condition[diag]:
                        end = np.array([start_row + one_up, start_col +\
                            col_change[diag]])
                        move = np.array([start, end])

                        # If a pawn is in its starting row, a two-up move is
                        # available as well
                        if start_row + one_up == 7 or start_row + one_up == 0:
                            available_nodes += self.pawn_promotion(move)
                        else:
                            take_square = self.board[start_row + one_up,\
                                start_col + col_change[diag]]

                            if (take_square != EMPTY_3).all() and\
                                    take_square[0] != self.side and\
                                    take_square[2] != 4 or (np.array([\
                                    start_row + one_up, start_col +\
                                    col_change[diag]]) == self.en_passant)\
                                    .all():
                                node = self.node_do_move(move)

                                if not node.is_checked(opp):
                                    available_nodes.append(node)

            # If the piece is a Bishop or Queen
            if piece == 2 or piece == 3:
                row_change = 1
                col_change = 1

                # Bishop move valid WHILE coordinates in range + square empty
                for _ in range(4):
                    row = start_row + row_change
                    col = start_col + col_change

                    while row in range(0, 8) and col in range(0, 8):
                        end = np.array([row, col])
                        move = np.array([start, end])
                        
                        # If the square is empty
                        if (self.board[row, col] == EMPTY_3).all():
                            node = self.node_do_move(move)

                            if not node.is_checked(opp):
                                available_nodes.append(node)

                        # If the square is occupied
                        else:
                            # If the square is takeable
                            if self.board[row, col][0] != self.side and\
                                    self.board[row, col][2] != 4:
                                node = self.node_do_move(move)

                                if not node.is_checked(opp):
                                    available_nodes.append(node)

                            break

                        row += row_change
                        col += col_change

                    row_change, col_change = col_change, row_change
                    col_change *= -1

        # Update on Heuristic Part 1: If someone is checkmated
        if len(available_nodes) == 0:
            # If there are no available nodes and checked, checkmate
            if self.is_checked(self.side):
                self.outcome = "Checkmate"

                self.h_value += HEUR_P1_ADD if self.side != self.player \
                    else -1 * HEUR_P1_ADD
            # But if you aren't checked, it's just a draw
            else:
                self.outcome = "Draw"

                self.h_value = int(self.h_value // HEUR_P1_DIVIDE)

        return available_nodes

    # Returns a node corresponding to a new (valid) move
    # Also updates the h-value and targeting matrices for the child
    def node_do_move(self, move):
        new_board = deepcopy(self.board)
        opp = 1 if self.side == 0 else 0
        additive = 1 if self.side == self.player else -1

        # Coordinates of piece that is about to move
        start_row = move[0, 0]
        start_col = move[0, 1]
        start_tup = new_board[start_row, start_col]
        start_id = start_tup[1]
        start_piece = start_tup[2]

        # Coordinates of square the piece is moving to
        end_row = move[1, 0]
        end_col = move[1, 1]
        end_tup = new_board[end_row, end_col]

        # Variables of the node to be returned
        new_my_squares = deepcopy(self.my_squares)
        new_my_moved = deepcopy(self.my_moved)
        new_my_targeted = deepcopy(self.my_targeted)
        new_en_passant = np.array([-1, -1])

        new_opp_squares = deepcopy(self.opp_squares)
        new_opp_moved = deepcopy(self.opp_moved)
        new_opp_targeted = deepcopy(self.opp_targeted)

        new_h_value = self.h_value
        # "Reset" Heuristic Part 4 values in order to update them
        new_h_value += additive * self.num_attacked(new_my_squares,\
            new_opp_targeted)
        new_h_value -= additive * self.num_attacked(new_opp_squares,\
            new_my_targeted)

        # Move piece from its original position
        self.alter_targeted(self.side, move[0], new_my_targeted, new_board, 0)
        self.alter_targeted(opp, move[0], new_opp_targeted, new_board, 0)
        self.update_targeted(move[0], new_my_targeted, new_board, 1)

        # Update both targeted matrices for when piece is moved
        if (end_tup == EMPTY_3).all():
            self.alter_targeted(self.side, move[1], new_my_targeted,\
                new_board, 1)
            self.alter_targeted(opp, move[1], new_opp_targeted, new_board, 1)

        # If an enemy piece was taken
        if (end_tup != EMPTY_3).all():
            end_id = end_tup[1]
            end_piece = end_tup[2]

            # Update on Heuristic Part 2: Piece point total
            new_h_value += additive * POINTS[end_piece]

            # Remove the enemy piece from targeted
            self.update_targeted(move[1], new_opp_targeted, new_board, 1)

            # Update opp variables
            new_opp_squares[end_id] = np.array([-1, -1])
            new_opp_moved[end_id] = True

            # Remove the enemy piece from the board
            new_board[end_row, end_col] = np.array(EMPTY_3)
        # Account for en passant
        elif (start_piece == 5 and move[1] == self.en_passant).all():
            one_down = -1 if self.side == 0 else 1

            end_tup = new_board[end_row + one_down, end_col]
            end_id = end_tup[1]
            end_piece = end_tup[2]

            # Update on Heuristic Part 2: Piece point total
            new_h_value += additive * POINTS[end_piece]

            # Update opp variables
            new_opp_squares[end_id] = np.array([-1, -1])
            new_opp_moved[end_id] = True

        # Account for castle
        elif start_piece == 4 and abs(start_col - end_col) == 2:
            # Left-side castle
            if end_col == 2:
                # Move the Rook too
                new_board[start_row, 0] = np.array(EMPTY_3)
                new_board[start_row, 3] = np.array([self.side, 0, 0])

                new_my_squares[0] = np.array([start_row, 3])
                new_my_moved[0] = True
            # Right-side castle
            else:
                # Move the Rook too
                new_board[start_row, 7] = np.array(EMPTY_3)
                new_board[start_row, 5] = np.array([self.side, 7, 0])

                new_my_squares[7] = np.array([start_row, 3])
                new_my_moved[7] = True

            # Update on Heuristic Part 6: If someone castles
            new_h_value += additive * HEUR_P6_ADD

        # Open en passant for opponent after Pawn double-up
        elif start_piece == 5 and abs(start_row - end_row) == 2:
            one_down = -1 if self.side == 0 else 1

            new_en_passant = np.array([start_row + one_down, start_col])

        # Swap the starting and ending squares (in order to move piece to the
        # new square and empty the old square)
        new_board[end_row, end_col] = new_board[start_row, start_col]
        new_board[start_row, start_col] = np.array([-1, -1, -1])

        # Add the just-moved piece to targeted
        self.update_targeted(move[1], new_my_targeted, new_board, 0)

        # Update my_moved
        new_my_squares[start_id] = np.array([end_row, end_col])

        # Update on Heuristic Part 3: How many times one's pieces are attacked
        new_h_value -= additive * self.num_attacked(new_my_squares,\
            new_opp_targeted)
        new_h_value += additive * self.num_attacked(new_opp_squares,\
            new_my_targeted)

        # Update on Heuristic Part 4: If a piece is moved from its starting
        # position
        if not new_my_moved[start_id]:
            new_h_value += additive * HEUR_P4_POINTS[start_piece]

            new_my_moved[start_id] = True

        # Prepare the node to be returned (swap [my <-> opp])
        ret_node = Node(new_board, opp, self.player)

        ret_node.my_squares = new_opp_squares
        ret_node.my_moved = new_opp_moved
        ret_node.my_targeted = new_opp_targeted
        ret_node.en_passant = new_en_passant

        ret_node.opp_squares = new_my_squares
        ret_node.opp_moved = new_my_moved
        ret_node.opp_targeted = new_my_targeted

        # Update on Heuristic Part 5: If opp is Checked
        # If enemy king is checked, heuristic++
        if ret_node.is_checked(opp):
            new_h_value += additive * HEUR_P5_ADD

        ret_node.h_value = new_h_value
        ret_node.move = move

        return ret_node

    # Returns all nodes corresponding a pawn promotion (Pawn ->
    # Rook/Knight/Bishop/Queen)
    def pawn_promotion(self, move):
        nodes = []

        new_board = deepcopy(self.board)
        opp = 1 if self.side == 0 else 0
        additive = 1 if self.side == self.player else -1

        # Coordinates of piece that is about to move
        start_row = move[0, 0]
        start_col = move[0, 1]
        start_tup = new_board[start_row, start_col]
        start_id = start_tup[1]
        start_piece = start_tup[2]

        # Coordinates of square the piece is moving to
        end_row = move[1, 0]
        end_col = move[1, 1]
        end_tup = new_board[end_row, end_col]

        # Variables of the node to be returned
        new_my_squares = deepcopy(self.my_squares)
        new_my_moved = deepcopy(self.my_moved)
        new_my_targeted = deepcopy(self.my_targeted)

        new_opp_squares = deepcopy(self.opp_squares)
        new_opp_moved = deepcopy(self.opp_moved)
        new_opp_targeted = deepcopy(self.opp_targeted)

        new_h_value = self.h_value
        # "Reset" Heuristic Part 4 values in order to update them
        new_h_value += additive * self.num_attacked(new_my_squares,\
            new_opp_targeted)
        new_h_value -= additive * self.num_attacked(new_opp_squares,\
            new_my_targeted)

        # Move piece from its original position
        self.alter_targeted(self.side, move[0], new_my_targeted, new_board, 0)
        self.alter_targeted(opp, move[0], new_opp_targeted, new_board, 0)
        self.update_targeted(move[0], new_my_targeted, new_board, 1)

        # Update both targeted matrices for when piece is moved
        if (end_tup == EMPTY_3).all():
            self.alter_targeted(self.side, move[1], new_my_targeted,\
                new_board, 1)
            self.alter_targeted(opp, move[1], new_opp_targeted, new_board, 1)

        # If an enemy piece was taken
        if (end_tup != EMPTY_3).all():
            end_id = end_tup[1]
            end_piece = end_tup[2]

            # Update on Heuristic Part 2: Piece point total
            new_h_value += additive * POINTS[end_piece]

            # Remove the enemy piece from targeted
            self.update_targeted(move[1], new_opp_targeted, new_board, 1)

            # Update opp variables
            new_opp_squares[end_id] = np.array([-1, -1])
            new_opp_moved[end_id] = True

            # Remove the enemy piece from the board
            new_board[end_row, end_col] = np.array(EMPTY_3)

        # Swap the starting and ending squares (in order to move piece to the
        # new square and empty the old square)
        new_board[end_row, end_col] = new_board[start_row, start_col]
        new_board[start_row, start_col] = np.array([-1, -1, -1])

        # Update my_moved
        new_my_squares[start_id] = np.array([end_row, end_col])

        # Cover all possible pawn promotion choices
        choices = np.array(list(range(4)))

        for choice in choices:
            newest_board = deepcopy(new_board)
            newest_board[end_row, end_col][2] = choices[choice]

            newest_my_targeted = deepcopy(new_my_targeted)

            newest_h_value = new_h_value

            self.update_targeted(move[1], newest_my_targeted, newest_board, 0)

            # Update on Heuristic Part 3: How many times one's pieces are attacked
            newest_h_value -= additive * self.num_attacked(new_my_squares,\
                new_opp_targeted)
            newest_h_value += additive * self.num_attacked(new_opp_squares,\
                newest_my_targeted)

            # Prepare the node to be returned (swap [my <-> opp])
            node = Node(newest_board, opp, self.player)

            node.my_squares = deepcopy(new_opp_squares)
            node.my_moved = deepcopy(new_opp_moved)
            node.my_targeted = deepcopy(new_opp_targeted)

            node.opp_squares = deepcopy(new_my_squares)
            node.opp_moved = deepcopy(new_my_moved)
            node.opp_targeted = newest_my_targeted

            # Update on Heuristic Part 5: If opp is Checked
            # If enemy king is checked, heuristic++
            if node.is_checked(opp):
                newest_h_value += additive * HEUR_P5_ADD

            node.h_value = newest_h_value
            node.move = move

            nodes.append(node)

        return nodes
