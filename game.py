import numpy as np
import math
import random

ROW_COUNT = 6
COL_COUNT = 7
PLAYER_PIECE = 1
AI_PIECE = 2

# Game Logic
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

# Minimax
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COL_COUNT//2])]
    score += center_array.count(piece) * 3

    # Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COL_COUNT - 3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    # Vertical
    for c in range(COL_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # Slop Up
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Slop Down
    for c in range(COL_COUNT - 3):
        for r in range(3, ROW_COUNT):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def get_valid_locations(board):
    valid_locations = []
    for c in range(COL_COUNT):
        if is_valid_location(board, c):
            valid_locations.append(c)
    return valid_locations

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, PLAYER_PIECE):
                return None, -1e10
            elif winning_move(board, AI_PIECE):
                return None, 1e10
            else: # Draw
                return None, 0
        else:
            return None, score_position(board, AI_PIECE)

    if maximizing_player: # AI
        value = -math.inf
        best_c = random.choice(valid_locations)
        for c in valid_locations:
            r = get_next_open_row(board, c)
            b_copy = board.copy()
            drop_piece(b_copy, r, c, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_c = c
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_c, value

    else: # Player
        value = math.inf
        best_c = random.choice(valid_locations)
        for c in valid_locations:
            r = get_next_open_row(board, c)
            b_copy = board.copy()
            drop_piece(b_copy, r, c, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_c = c
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_c, value

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
        piece = AI_PIECE
        ''' Player 2 as human
        try:
            selection = int(input(f'Player {piece}, Select an column(0-{COL_COUNT-1}): '))
            if selection < 0 or selection >= COL_COUNT:
                print(f'Invalid column, please enter a number between 0 and {COL_COUNT-1}.')
                continue

        except ValueError:
            print('Invalid input, please enter a number between 0 and 6.')
            continue
        '''
        selection, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
        print(f'AI selected column {selection} with score {minimax_score}.')

    if is_valid_location(board, selection):
        row = get_next_open_row(board, selection)
        drop_piece(board, row, selection, piece)
        if winning_move(board, piece):
            print_board(board)
            if piece == PLAYER_PIECE:
                print("PLAYER WINS!")
            else:
                print("AI WINS!")
            game_over = True

        elif len(board[board == 0]) == 0: # boolean indexing, return all elements that =0 in the array
            print("DRAW!")
            game_over = True

        print_board(board)
        turn = (turn + 1) % 2

    else:
        print("Column is full! Please choose another one.")