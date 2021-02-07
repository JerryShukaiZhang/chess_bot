Chess Bot (Artificial Intelligence Project)
Jerry Shukai Zhang
12/06/2020 - 02/03/2021 (Untested Changes)

*** En passant/pawn promotion/castling have been implemented but not tested. ***

Returns next best move in Chess game using Minimax algorithm with Alpha-Beta pruning and tree-like search. Heuristic was independently conceived.

Pieces represented in array of 3 ints [Side, ID, Piece]
    Side: 0 = White, 1 = Black
    ID: Number for each piece used for indexing purposes
    Piece: 0 = Rook, 1 = Knight, 2 = Bishop, 3 = Queen, 4 = King, 5 = Pawn

Moves represented in array of 2 coordinates [[old_row, old_col], [new_row, new_col]]
Note that [row, col] = [y, x], so coordinates will be flipped
Coordinates are referred to as "squares"

Points going to the computer get added to the heuristic value; points going to the computer's opponent are subtracted from the heuristic value

Heuristic:
    Part 1 = +100 to the player who checkmates
    Part 2 = Piece Point Total (Rook = 5, Knight/Bishop = 3, Queen = 9, Pawn = 1)
    Part 3 = Number of times that one's pieces are attacked
    Part 4 = Number of pieces moved from starting positions
    Part 5 = +3 to a player who checks
    Part 6 = +3 to a player that castles

After the Minimax algorithm completes the maximum number of look-aheads, the heuristic value is used to choose between moves.

Castling is represented as a move from a King two spaces away from its original position (i.e E1 to C1).

Currently, to return a move in reasonable time, the bot has been set to look only 4 moves ahead.
Also, the moves_made field of SearchTree is not currently being used.

There is no check for illegal moves.