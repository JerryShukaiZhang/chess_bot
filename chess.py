"""
User-Interacting file for Chess Bot
Jerry Shukai Zhang
12/06/2020 - 02/03/2021
"""

from searchtree import SearchTree
import numpy as np

PIECES = ["Ro", "Kn", "Bi", "Qu", "Ki", "Pa"]
# LETTERS used for printing the columns of the chess board
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
EMPTY = np.array([-1, -1, -1])


# Keeps taking inputs from the user until a letter in acceptable_letters
# is input
def get_valid_letter(acceptable_letters):
    letter = input().strip().upper()

    while letter not in acceptable_letters or len(letter) > 1:
        print("I did not understand that. Please enter a valid letter (",\
            end="")

        for letter in range(len(acceptable_letters)):
            print(acceptable_letters[letter], end="")

            if letter != len(acceptable_letters) - 1:
                print(", ", end="")

        print("): ", end="")

        letter = input().strip().upper()

    return letter


# Takes an input from the user until a "valid" coordinate is entered
# A valid coordinate is of length 2, has a chess column in its first index
# a number from 1 to 8 in its second index
def get_valid_coord():
    coord = input().strip().upper()

    while len(coord) != 2 or\
            (not coord[0].isalpha()) or (coord[0] not in LETTERS) or\
            (not coord[1].isnumeric()) or (int(coord[1]) not in range(1, 9)):
        print("I did not understand that. Please enter a valid Column/Row"
            " (i.e \"B6\"): ", end="")

        coord = input().strip().upper()

    return coord


# Takes in inputs from the user to execute the next move
# chess_game = the SearchTree holding the current game
def take_move(chess_game):
    print("Please enter the Column/Row of your piece: ", end="")
    old_coord = get_valid_coord()

    old_col = old_coord[0].upper()
    old_col = LETTERS.index(old_col)

    old_row = int(old_coord[1]) - 1

    old = np.array([old_row, old_col])

    print("Next, the Column/Row you want to move to : ", end="")
    new_coord = get_valid_coord()

    new_col = new_coord[0].upper()
    new_col = LETTERS.index(new_col)

    new_row = int(new_coord[1]) - 1

    new = np.array([new_row, new_col])

    move = np.array([old, new])

    chess_game.tree_do_move(move)


def main():
    # Set up the game
    print("What side are you playing? (W/B): ", end="")

    player_side = get_valid_letter("WB")
    player_side = 0 if player_side == "W" else 1

    chess_game = SearchTree(player_side)

    # Run the game
    while chess_game.curr_board.outcome == "Ongoing":
        # Print state of game
        print("\nNow, the board looks like: ")

        # Print col numbers
        row_num = [" "]
        for i in range(8):
            row_num.append(" " + LETTERS[i] + " ")

        # Print the pieces of the board
        print_board = [row_num]
        for row in range(7, -1, -1):
            new_row = [str(row + 1)]

            for col in range(8):
                new_str = ""

                # If the square is empty, print an appropriate "empty" string
                if (chess_game.curr_board.board[row, col] == EMPTY).all():
                    new_str = "   "
                # Otherwise, print the piece in the square
                else:
                    # Print the side of the piece
                    new_str += "W" if chess_game.curr_board.board[row, col]\
                        [0] == 0 else "B"
                    # Print the type of piece
                    new_str += PIECES[chess_game.curr_board.board[row, col]\
                        [2]]
                new_row.append(new_str)
            print_board.append(new_row)

        for row in print_board:
            print(row)
        print()

        # If it is the opponent's turn
        if chess_game.curr_board.side != chess_game.player:
            take_move(chess_game)
        # Otherwise, if it is the computer's turn
        else:
            # Return the suggested move
            suggested_move = chess_game.find_next_move()

            old_row = suggested_move[0, 0] + 1
            old_col = LETTERS[suggested_move[0, 1]]

            new_row = suggested_move[1, 0] + 1
            new_col = LETTERS[suggested_move[1, 1]]

            print("Suggested move: ", (old_col, old_row), " to ", (new_col,\
                new_row), "; Do this? (Y/N): ", end="")
            move = get_valid_letter("YN")

            if move == "Y":
                chess_game.tree_do_move(suggested_move)
            else:
                take_move(chess_game)

    # Print the outcome of the game
    if chess_game.curr_board.outcome == "Draw":
        print("The game was a draw!")
    # If the game was a checkmate
    else:
        if chess_game.curr_board.side == chess_game.player:
            print("You won!")
        else:
            print("You lost :(")


if __name__ == "__main__":
    main()