import time
import math
import random
import threading
import torch
import pygame

from game import (
    get_valid_locations,
    get_immediate_winning_move,
    minimax,
    PLAYER_PIECE,
    AI_PIECE,
)
from agent import DuelingConnect4Net
from benchmark import network_move

# Custom Pygame event for asynchronous AI move completion
AI_MOVE_EVENT = pygame.USEREVENT + 1


class BaseAIPlayer:
    def __init__(self, my_piece, opp_piece, name="AI"):
        self.my_piece = my_piece
        self.opp_piece = opp_piece
        self.name = name

    def get_move(self, board):
        raise NotImplementedError


class DQNPlayer(BaseAIPlayer):
    def __init__(self, my_piece=AI_PIECE, opp_piece=PLAYER_PIECE, model_path="model/connect4_model_selfplay.pth"):
        super().__init__(my_piece, opp_piece, name="DQN 神经网络 AI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DuelingConnect4Net().to(self.device)
        # A missing/incompatible checkpoint must not silently play random weights.
        self.net.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.net.eval()

    def get_move(self, board):
        return network_move(self.net, board, self.my_piece, self.opp_piece)


class MinimaxPlayer(BaseAIPlayer):
    def __init__(self, my_piece=AI_PIECE, opp_piece=PLAYER_PIECE, depth=4):
        depth_names = {2: "简单 (深度2)", 4: "中等 (深度4)", 5: "困难 (深度5)"}
        name_suffix = depth_names.get(depth, f"深度{depth}")
        super().__init__(my_piece, opp_piece, name=f"Minimax AI - {name_suffix}")
        self.depth = depth

    def get_move(self, board):
        valid_moves = get_valid_locations(board)
        if not valid_moves:
            return None

        # 1-ply fast check to avoid expensive search if forced
        win_move = get_immediate_winning_move(board, self.my_piece)
        if win_move is not None and win_move in valid_moves:
            return win_move

        block_move = get_immediate_winning_move(board, self.opp_piece)
        if block_move is not None and block_move in valid_moves:
            return block_move

        col, _ = minimax(
            board,
            self.depth,
            -math.inf,
            math.inf,
            maximizing_player=True,
            ai_piece=self.my_piece,
            player_piece=self.opp_piece,
        )
        if col is None or col not in valid_moves:
            col = random.choice(valid_moves)
        return col


class RandomPlayer(BaseAIPlayer):
    def __init__(self, my_piece=AI_PIECE, opp_piece=PLAYER_PIECE, tactical=True):
        super().__init__(my_piece, opp_piece, name="随机战术 AI" if tactical else "纯随机 AI")
        self.tactical = tactical

    def get_move(self, board):
        valid_moves = get_valid_locations(board)
        if not valid_moves:
            return None

        if self.tactical:
            win_move = get_immediate_winning_move(board, self.my_piece)
            if win_move is not None and win_move in valid_moves:
                return win_move

            block_move = get_immediate_winning_move(board, self.opp_piece)
            if block_move is not None and block_move in valid_moves:
                return block_move

        return random.choice(valid_moves)


class AsyncAIWorker:
    """
    Manages asynchronous AI calculation on a background thread.
    Prevents GUI freeze and uses generation tokens to discard stale results.
    """

    def __init__(self):
        self.current_gen_id = 0
        self.is_thinking = False
        self._thread = None

    def start_thinking(self, ai_player, board, min_delay=0.3):
        """
        Dispatches AI calculation on a daemon thread.
        min_delay gives a natural pace so fast AIs don't respond in 0.001s.
        """
        self.current_gen_id += 1
        gen_id = self.current_gen_id
        self.is_thinking = True

        board_copy = board.copy()

        def _task():
            start_time = time.time()
            best_col = ai_player.get_move(board_copy)
            elapsed = time.time() - start_time
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)

            # Post event to main thread via Pygame event queue
            event = pygame.event.Event(AI_MOVE_EVENT, {"col": best_col, "gen_id": gen_id})
            pygame.event.post(event)

        self._thread = threading.Thread(target=_task, daemon=True)
        self._thread.start()
        return gen_id

    def cancel(self):
        """Invalidates pending computations by incrementing generation ID."""
        self.current_gen_id += 1
        self.is_thinking = False
