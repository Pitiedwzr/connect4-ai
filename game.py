import numpy as np


ROW_COUNT = 6
COL_COUNT = 7

def create_board():
    board = np.zeros((ROW_COUNT, COL_COUNT), dtype=int)
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    if board[ROW_COUNT-1][col] != 0:
        return False
    else:
        return True

def get_next_open_row(board, col):
    for row in range(ROW_COUNT):
        if board[row][col] == 0:
            return row
    return None


board = create_board()
game_over = False
turn = 0

while not game_over:
    if turn == 0:
        selection = int(input('Player 1, Select an column(0-6): '))
        if is_valid_location(board, selection):
            row = get_next_open_row(board, selection)
            drop_piece(board, row, selection, 1)
    else:
        selection = int(input('Player 2, Select an column(0-6): '))
        if is_valid_location(board, selection):
            row = get_next_open_row(board, selection)
            drop_piece(board, row, selection, 2)

    print(np.flip(board, 0))

    turn += 1
    turn %= 2