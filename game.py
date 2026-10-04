import numpy as np
import math
import random
import torch
from agent import DuelingConnect4Net

ROW_COUNT = 6
COL_COUNT = 7
PLAYER_PIECE = 1
AI_PIECE = 2
CURRENT_AI = "DQN" # Minimax or DQN
DQN_MODEL_PATH = "model/connect4_model.pth"

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

def get_winning_coordinates(board, piece):
    """
    Finds the 4 coordinates [(r, c), ...] that form a 4-in-a-row for piece.
    Returns None if no winning line exists.
    """
    # Horizontal
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board[r][c+i] == piece for i in range(4)):
                return [(r, c+i) for i in range(4)]

    # Vertical
    for r in range(ROW_COUNT - 3):
        for c in range(COL_COUNT):
            if all(board[r+i][c] == piece for i in range(4)):
                return [(r+i, c) for i in range(4)]

    # Slope Up
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return [(r+i, c+i) for i in range(4)]

    # Slope Down
    for c in range(COL_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return [(r-i, c+i) for i in range(4)]

    return None

def get_immediate_winning_move(board, piece):
    for c in get_valid_locations(board):
        r = get_next_open_row(board, c)
        board[r][c] = piece
        is_win = winning_move(board, piece)
        board[r][c] = 0
        if is_win:
            return c
    return None

def is_suicide_move(board, col, my_piece, opp_piece):
    row = get_next_open_row(board, col)
    if row is None:
        return False

    if row + 1 >= ROW_COUNT:
        return False

    board[row][col] = my_piece

    board[row + 1][col] = opp_piece

    opp_wins = winning_move(board, opp_piece)

    board[row + 1][col] = 0
    board[row][col] = 0

    return opp_wins

# Minimax
def evaluate_window(window, piece, opp_piece=None):
    score = 0
    if opp_piece is None:
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

def score_position(board, piece, opp_piece=None):
    score = 0
    if opp_piece is None:
        opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    center_array = [int(i) for i in list(board[:, COL_COUNT//2])]
    score += center_array.count(piece) * 3

    # Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COL_COUNT - 3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece, opp_piece)

    # Vertical
    for c in range(COL_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece, opp_piece)

    # Slop Up
    for c in range(COL_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece, opp_piece)

    # Slop Down
    for c in range(COL_COUNT - 3):
        for r in range(3, ROW_COUNT):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece, opp_piece)

    return score

def get_valid_locations(board):
    valid_locations = []
    for c in range(COL_COUNT):
        if is_valid_location(board, c):
            valid_locations.append(c)
    return valid_locations

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizing_player, ai_piece=AI_PIECE, player_piece=PLAYER_PIECE):
    valid_locations = get_valid_locations(board)
    is_terminal = winning_move(board, player_piece) or winning_move(board, ai_piece) or len(valid_locations) == 0

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, player_piece):
                return None, -1e10
            elif winning_move(board, ai_piece):
                return None, 1e10
            else: # Draw
                return None, 0
        else:
            return None, score_position(board, ai_piece, player_piece)

    if maximizing_player: # AI
        value = -math.inf
        best_c = random.choice(valid_locations)
        for c in valid_locations:
            r = get_next_open_row(board, c)
            b_copy = board.copy()
            drop_piece(b_copy, r, c, ai_piece)
            new_score = minimax(b_copy, depth-1, alpha, beta, False, ai_piece, player_piece)[1]
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
            drop_piece(b_copy, r, c, player_piece)
            new_score = minimax(b_copy, depth-1, alpha, beta, True, ai_piece, player_piece)[1]
            if new_score < value:
                value = new_score
                best_c = c
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_c, value

# DQN use
def get_state_tensor(board, ai_piece, opponent_piece):
    """
    Converts a 6x7 numpy board into a 2-channel PyTorch tensor.
    Returns shape: [1, 2, 6, 7]
    """
    ai_channel = (board == ai_piece).astype(np.float32)
    op_channel = (board == opponent_piece).astype(np.float32)
    stacked_channels = np.stack([ai_channel, op_channel])
    return torch.tensor(stacked_channels).unsqueeze(0)

# Main logic
if __name__ == "__main__":
    board = create_board()
    game_over = False
    turn = 0

    if CURRENT_AI == "DQN":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dqn_ai = DuelingConnect4Net().to(device)
        dqn_ai.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
        dqn_ai.eval()

    first_choice = input("Do you want to play first? (y/n, default y): ").strip().lower()
    turn = 0 if first_choice != 'n' else 1

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
            valid_moves = get_valid_locations(board)
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
            if CURRENT_AI == "Minimax":
                selection, score = minimax(board, 5, -math.inf, math.inf, True)
                print(f'Minimax selected column {selection} with score {score}.')
            elif CURRENT_AI == "DQN":
                # 1-ply tactical check: immediate win or block
                win_c = get_immediate_winning_move(board, AI_PIECE)
                block_c = get_immediate_winning_move(board, PLAYER_PIECE)
                if win_c is not None:
                    selection = win_c
                    print(f'DQN selected column {selection} (immediate winning move).')
                elif block_c is not None:
                    selection = block_c
                    print(f'DQN selected column {selection} (immediate blocking move).')
                else:
                    safe_moves = [c for c in valid_moves if not is_suicide_move(board, c, AI_PIECE, PLAYER_PIECE)]

                    candidate_moves = safe_moves if len(safe_moves) > 0 else valid_moves

                    state_tensor = get_state_tensor(board, AI_PIECE, PLAYER_PIECE).to(device)

                    with torch.no_grad():
                        q_value = dqn_ai(state_tensor)[0].cpu().numpy()

                    max_q = -math.inf
                    selection = candidate_moves[0]
                    for c in candidate_moves:
                        if q_value[c] > max_q:
                            max_q = q_value[c]
                            selection = c

                    print(f'DQN selected column {selection} with expected reward {max_q:.3f}.')

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