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
DQN_MODEL_PATH = "model/connect4_model_selfplay.pth"

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
    moves = get_winning_moves(board, piece)
    return moves[0] if moves else None


# Seven bits per column: six playable cells and a sentinel separating columns.
def _piece_bits(board, piece):
    return sum(1 << (7 * c + r) for r, c in zip(*np.where(board == piece)))


def _has_four(bits):
    for shift in (1, 7, 6, 8):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def _winning_columns(bits, heights):
    return [c for c in (3, 2, 4, 1, 5, 0, 6)
            if heights[c] < ROW_COUNT
            and _has_four(bits | (1 << (7 * c + heights[c])))]


def get_winning_moves(board, piece):
    """All immediately winning columns, without mutating the board."""
    heights = np.count_nonzero(board, axis=0).tolist()
    return _winning_columns(_piece_bits(board, piece), heights)


def get_candidate_moves(board, my_piece, opp_piece):
    """Shared DQN policy: center opening, wins, immediate safety, fork safety.

    Reject an opponent reply that creates two winning columns unless we can
    win immediately after that reply. If every move loses, retain legal moves.
    This bounded tactical search does not prove safety against deeper traps.
    """
    heights = np.count_nonzero(board, axis=0).tolist()
    legal = [c for c in (3, 2, 4, 1, 5, 0, 6) if heights[c] < ROW_COUNT]
    if not legal:
        return []
    if not any(heights):
        return [COL_COUNT // 2]
    mine = _piece_bits(board, my_piece)
    theirs = _piece_bits(board, opp_piece)
    wins = _winning_columns(mine, heights)
    if wins:
        return wins

    safe, fork_safe = [], []
    for c in legal:
        after_mine = mine | (1 << (7 * c + heights[c]))
        heights[c] += 1
        if not _winning_columns(theirs, heights):
            safe.append(c)
            allows_fork = False
            for reply in legal:
                if heights[reply] >= ROW_COUNT:
                    continue
                after_theirs = theirs | (1 << (7 * reply + heights[reply]))
                heights[reply] += 1
                if (not _winning_columns(after_mine, heights)
                        and len(_winning_columns(after_theirs, heights)) >= 2):
                    allows_fork = True
                heights[reply] -= 1
                if allows_fork:
                    break
            if not allows_fork:
                fork_safe.append(c)
        heights[c] -= 1
    return fork_safe or safe or legal

def is_suicide_move(board, col, my_piece, opp_piece):
    """Whether this move allows any immediate winning opponent reply."""
    row = get_next_open_row(board, col)
    if row is None:
        return True
    heights = np.count_nonzero(board, axis=0).tolist()
    mine = _piece_bits(board, my_piece) | (1 << (7 * col + row))
    if _has_four(mine):
        return False
    heights[col] += 1
    return bool(_winning_columns(_piece_bits(board, opp_piece), heights))

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
    valid_locations = sorted(get_valid_locations(board), key=lambda c: abs(c - COL_COUNT // 2))
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
                candidate_moves = get_candidate_moves(board, AI_PIECE, PLAYER_PIECE)
                state_tensor = get_state_tensor(board, AI_PIECE, PLAYER_PIECE).to(device)
                with torch.no_grad():
                    q_values = dqn_ai(state_tensor)[0].cpu().tolist()
                selection = max(candidate_moves, key=lambda c: q_values[c])
                print(f'DQN selected column {selection} with estimated return {q_values[selection]:.3f}.')

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
