import numpy as np


ROW_COUNT = 6
COL_COUNT = 7
PLAYER_PIECE = 1
AI_PIECE = 2

def create_board():
    return np.zeros((ROW_COUNT, COL_COUNT), dtype=int)

def print_board(board):
    print(np.flip(board, 0))

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
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    return False

# Main logic
board = create_board()
game_over = False
turn = 0

print_board(board)

while not game_over:
    piece = 0
    if turn == 0:
        piece = PLAYER_PIECE
        try:
            selection = int(input(f'Player {piece}, Select an column(0-{COL_COUNT-1}): '))
            if selection < 0 or selection >= COL_COUNT:
                print(f'Invalid column, please enter a number between 0 and {COL_COUNT-1}.')
                continue

        except ValueError:
            print('Invalid input, please enter a number between 0 and 6.')
            continue

    else:
        piece = AI_PIECE # Leave for future implement
        try:
            selection = int(input(f'Player {piece}, Select an column(0-{COL_COUNT-1}): '))
            if selection < 0 or selection >= COL_COUNT:
                print(f'Invalid column, please enter a number between 0 and {COL_COUNT-1}.')
                continue

        except ValueError:
            print('Invalid input, please enter a number between 0 and 6.')
            continue

    if is_valid_location(board, selection):
        row = get_next_open_row(board, selection)
        drop_piece(board, row, selection, piece)
        if winning_move(board, piece):
            print_board(board)
            print(f"PLAYER {piece} WINS!")
            game_over = True

        elif len(board[board == 0]) == 0:
            print("DRAW!")
            game_over = True

        print_board(board)
        turn = (turn + 1) % 2

    else:
        print("Column is full! Please choose another one.")