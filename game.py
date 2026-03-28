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

def winning_move(board, piece):
    # Horizontal
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Vertical
    for r in range(ROW_COUNT - 3):
        for c in range(COL_COUNT):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Slop Up
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Slop Down
    for c in range(COL_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r+1][c-1] == piece and board[r+2][c-2] == piece and board[r+3][c-3] == piece:
                return True

    return False

board = create_board()
game_over = False
turn = 0

while not game_over:
    if turn == 0:
        selection = int(input('Player 1, Select an column(0-6): '))
        if is_valid_location(board, selection):
            row = get_next_open_row(board, selection)
            drop_piece(board, row, selection, 1)
            if winning_move(board, 1):
                print("PLAYER 1 WINS!!")
                game_over = True
    else:
        selection = int(input('Player 2, Select an column(0-6): '))
        if is_valid_location(board, selection):
            row = get_next_open_row(board, selection)
            drop_piece(board, row, selection, 2)
            if winning_move(board, 2):
                print("PLAYER 2 WINS!!")
                game_over = True

    print(np.flip(board, 0))
    turn += 1
    turn %= 2