"""
Connect 4 AI - Modern Pygame GUI with Real-time Win Rate Evaluation
Features:
- Crisp High-DPI Awareness & Anti-Aliased Graphics (Zero Blur)
- Real-time win rate evaluation bar (DQN Neural Network & Heuristic Dual-Engine)
- Multiple opponent types: DQN, Minimax (Depth 2/4/5), Tactical Random, 2-Player Pass & Play
- Smooth 60 FPS piece dropping physics animation
- Asynchronous AI computation (non-blocking, zero GUI stutter)
- Move history, Smart Undo, Reset, and AI vs AI Spectator Mode
"""

import sys
import math
import time
import ctypes
import pygame
import pygame.gfxdraw
import numpy as np

# --- 1. Windows High-DPI Awareness (Fixes System Scaling Blur) ---
try:
    # Per-Monitor DPI aware (V2)
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        # Fallback to system-level DPI aware
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from game import (
    create_board,
    drop_piece,
    is_valid_location,
    get_next_open_row,
    winning_move,
    is_terminal_node,
    get_winning_coordinates,
    ROW_COUNT,
    COL_COUNT,
    PLAYER_PIECE,
    AI_PIECE,
)
from evaluator import WinRateEvaluator
from ai_player import DQNPlayer, MinimaxPlayer, RandomPlayer, AsyncAIWorker, AI_MOVE_EVENT

# --- Pygame Initialization ---
pygame.init()
pygame.display.set_caption("Connect 4 AI - Intelligent Agent Battle")

# Window Configuration
SCREEN_WIDTH = 1140
SCREEN_HEIGHT = 750
FPS = 60

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# --- Crisp Modern Typography ---
available_fonts = pygame.font.get_fonts()
if "segoeui" in available_fonts:
    FONT_FAMILY = "segoeui"
elif "sfprodisplay" in available_fonts:
    FONT_FAMILY = "sfprodisplay"
elif "arial" in available_fonts:
    FONT_FAMILY = "arial"
else:
    FONT_FAMILY = "freesansbold"

FONT_TITLE = pygame.font.SysFont(FONT_FAMILY, 24, bold=True)
FONT_SUBTITLE = pygame.font.SysFont(FONT_FAMILY, 14)
FONT_BOLD = pygame.font.SysFont(FONT_FAMILY, 17, bold=True)
FONT_NORMAL = pygame.font.SysFont(FONT_FAMILY, 15)
FONT_SMALL = pygame.font.SysFont(FONT_FAMILY, 13)
FONT_SCORE = pygame.font.SysFont(FONT_FAMILY, 14, bold=True)

# --- Color Palette (Modern Dark Theme) ---
COLOR_BG = (15, 23, 42)             # Slate 900
COLOR_CARD = (30, 41, 59)           # Slate 800
COLOR_CARD_BORDER = (51, 65, 85)    # Slate 700
COLOR_TEXT_MAIN = (248, 250, 252)   # Slate 50
COLOR_TEXT_MUTED = (148, 163, 184)  # Slate 400
COLOR_ACCENT = (56, 189, 248)       # Sky 400

# Board & Piece Colors
COLOR_BOARD_FRAME = (30, 58, 138)   # Deep Blue Frame
COLOR_BOARD_INNER = (37, 99, 235)   # Royal Blue Board
COLOR_SLOT_EMPTY = (11, 17, 32)      # Empty Slot Cavity
COLOR_RED = (239, 68, 68)            # Red (Player 1)
COLOR_RED_HIGHLIGHT = (252, 165, 165)
COLOR_YELLOW = (245, 158, 11)        # Amber Yellow (Player 2)
COLOR_YELLOW_HIGHLIGHT = (253, 230, 138)
COLOR_WIN_GLOW = (255, 255, 255)

# Button Colors
COLOR_BTN_DEFAULT = (51, 65, 85)
COLOR_BTN_HOVER = (71, 85, 105)
COLOR_BTN_ACTIVE = (37, 99, 235)
COLOR_BTN_DANGER = (220, 38, 38)
COLOR_BTN_DANGER_ACTIVE = (185, 28, 28)

# --- Layout Definitions ---
RECT_EVAL_BAR = pygame.Rect(32, 95, 46, 580)
RECT_BOARD_AREA = pygame.Rect(108, 95, 660, 580)
RECT_SIDEBAR = pygame.Rect(796, 25, 318, 700)


def draw_aa_circle(surface, x, y, radius, color):
    """Draws an anti-aliased smooth solid circle."""
    x_int, y_int, r_int = int(round(x)), int(round(y)), int(round(radius))
    if r_int <= 0:
        return
    pygame.gfxdraw.filled_circle(surface, x_int, y_int, r_int, color)
    pygame.gfxdraw.aacircle(surface, x_int, y_int, r_int, color)


class DropAnimation:
    """Manages smooth piece falling physics with natural bounce."""
    def __init__(self, col, target_row, piece, start_y, target_y):
        self.col = col
        self.target_row = target_row
        self.piece = piece
        self.current_y = start_y
        self.target_y = target_y
        self.velocity = 0.0
        self.gravity = 1.6
        self.bounces = 0
        self.is_done = False

    def update(self):
        if self.is_done:
            return
        self.velocity += self.gravity
        self.current_y += self.velocity
        if self.current_y >= self.target_y:
            self.current_y = self.target_y
            if self.bounces == 0:
                self.velocity = -self.velocity * 0.28
                self.bounces += 1
            else:
                self.is_done = True


class Connect4GUI:
    def __init__(self):
        self.board = create_board()
        self.turn = 0  # 0: Red (Piece 1), 1: Yellow (Piece 2)
        self.game_over = False
        self.winner = None
        self.win_coords = None
        self.move_history = []  # List of (row, col, piece)

        # Settings
        self.opponent_type = "DQN"     # "DQN", "Minimax", "Random", "Human"
        self.minimax_depth = 4         # 2, 4, 5
        self.first_mover = "Human"     # "Human" (Player is Red), "AI" (AI is Red)
        self.ai_vs_ai = False

        # Evaluator & Async AI Worker
        self.evaluator = WinRateEvaluator()
        self.ai_worker = AsyncAIWorker()

        # Dynamic Win Rate Smoothing (Lerp)
        self.display_p1_rate = 0.5
        self.target_p1_rate = 0.5
        self.eval_source = "DQN Model"

        # Animation & Hover State
        self.anim = None
        self.hover_col = None
        self.ui_buttons = []

        self.update_eval()

    def get_current_ai_player(self):
        """Builds AIPlayer instance based on game configuration and turn."""
        if self.ai_vs_ai:
            current_piece = PLAYER_PIECE if self.turn == 0 else AI_PIECE
            opp_piece = AI_PIECE if self.turn == 0 else PLAYER_PIECE
        else:
            if self.first_mover == "Human":
                current_piece = AI_PIECE
                opp_piece = PLAYER_PIECE
            else:
                current_piece = PLAYER_PIECE
                opp_piece = AI_PIECE

        if self.opponent_type == "DQN":
            return DQNPlayer(my_piece=current_piece, opp_piece=opp_piece)
        elif self.opponent_type == "Minimax":
            return MinimaxPlayer(my_piece=current_piece, opp_piece=opp_piece, depth=self.minimax_depth)
        elif self.opponent_type == "Random":
            return RandomPlayer(my_piece=current_piece, opp_piece=opp_piece, tactical=True)
        return None

    def is_ai_turn(self):
        if self.game_over:
            return False
        if self.ai_vs_ai:
            return True
        if self.opponent_type == "Human":
            return False
        if self.first_mover == "Human":
            return self.turn == 1  # AI is Yellow
        else:
            return self.turn == 0  # AI is Red

    def update_eval(self):
        current_piece = PLAYER_PIECE if self.turn == 0 else AI_PIECE
        p1, p2, src = self.evaluator.evaluate(self.board, current_piece)
        self.target_p1_rate = p1
        # Translate source label if necessary
        self.eval_source = "DQN Neural Net" if "DQN" in src else "Heuristic Engine"

    def reset_game(self):
        self.ai_worker.cancel()
        self.board = create_board()
        self.turn = 0
        self.game_over = False
        self.winner = None
        self.win_coords = None
        self.move_history.clear()
        self.anim = None
        self.update_eval()
        self.display_p1_rate = 0.5
        self.trigger_ai_if_needed()

    def undo_move(self):
        if self.ai_worker.is_thinking or not self.move_history:
            return

        self.anim = None
        self.ai_worker.cancel()

        if self.opponent_type != "Human" and not self.ai_vs_ai:
            # Smart Undo: rollback 2 plies if AI has responded, or 1 ply if game over / player pending
            if self.game_over:
                human_piece = PLAYER_PIECE if self.first_mover == "Human" else AI_PIECE
                steps = 1 if self.winner == human_piece else min(2, len(self.move_history))
            elif not self.is_ai_turn():
                steps = min(2, len(self.move_history))
            else:
                steps = 1

            for _ in range(steps):
                if self.move_history:
                    r, c, _ = self.move_history.pop()
                    self.board[r][c] = 0
                    self.turn = (self.turn - 1) % 2
        else:
            # Local 2-Player or AI vs AI: Undo single move
            if self.move_history:
                r, c, _ = self.move_history.pop()
                self.board[r][c] = 0
                self.turn = (self.turn - 1) % 2

        self.game_over = False
        self.winner = None
        self.win_coords = None
        self.update_eval()
        self.trigger_ai_if_needed()

    def make_move(self, col):
        if not is_valid_location(self.board, col) or self.game_over or self.anim is not None:
            return False

        row = get_next_open_row(self.board, col)
        piece = PLAYER_PIECE if self.turn == 0 else AI_PIECE

        # Start smooth falling animation
        slot_w = RECT_BOARD_AREA.width / COL_COUNT
        slot_h = (RECT_BOARD_AREA.height - 20) / ROW_COUNT
        start_y = RECT_BOARD_AREA.top - 25
        target_y = RECT_BOARD_AREA.bottom - (row + 0.5) * slot_h - 10
        self.anim = DropAnimation(col, row, piece, start_y, target_y)

        # Place piece on board state
        drop_piece(self.board, row, col, piece)
        self.move_history.append((row, col, piece))

        # Check terminal state
        if winning_move(self.board, piece):
            self.game_over = True
            self.winner = piece
            self.win_coords = get_winning_coordinates(self.board, piece)
        elif is_terminal_node(self.board):
            self.game_over = True
            self.winner = None

        self.turn = (self.turn + 1) % 2
        self.update_eval()
        return True

    def trigger_ai_if_needed(self):
        if self.is_ai_turn() and not self.ai_worker.is_thinking and self.anim is None:
            ai_player = self.get_current_ai_player()
            if ai_player:
                delay = 0.45 if self.ai_vs_ai else 0.2
                self.ai_worker.start_thinking(ai_player, self.board, min_delay=delay)

    def run(self):
        while True:
            # 1. Update Physics
            if self.anim:
                self.anim.update()
                if self.anim.is_done:
                    self.anim = None
                    self.trigger_ai_if_needed()

            # 2. Smooth Lerp for Win Rate Meter
            lerp_speed = 0.12
            self.display_p1_rate += (self.target_p1_rate - self.display_p1_rate) * lerp_speed

            # 3. Check AI vs AI Trigger
            if self.ai_vs_ai and self.is_ai_turn() and not self.ai_worker.is_thinking and self.anim is None:
                self.trigger_ai_if_needed()

            # 4. Handle Inputs & Events
            mouse_pos = pygame.mouse.get_pos()
            self.update_hover(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == AI_MOVE_EVENT:
                    self.ai_worker.is_thinking = False
                    if event.gen_id == self.ai_worker.current_gen_id and not self.game_over:
                        col = event.col
                        if col is not None:
                            self.make_move(col)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            # 5. Render Crisp Frame
            self.draw()
            pygame.display.flip()
            clock.tick(FPS)

    def update_hover(self, mouse_pos):
        if self.game_over or self.is_ai_turn() or self.anim is not None:
            self.hover_col = None
            return

        if RECT_BOARD_AREA.collidepoint(mouse_pos):
            col_w = RECT_BOARD_AREA.width / COL_COUNT
            col = int((mouse_pos[0] - RECT_BOARD_AREA.left) // col_w)
            if 0 <= col < COL_COUNT and is_valid_location(self.board, col):
                self.hover_col = col
            else:
                self.hover_col = None
        else:
            self.hover_col = None

    def handle_click(self, pos):
        # Click on Board Columns
        if RECT_BOARD_AREA.collidepoint(pos) and not self.is_ai_turn() and not self.game_over and self.anim is None:
            col_w = RECT_BOARD_AREA.width / COL_COUNT
            col = int((pos[0] - RECT_BOARD_AREA.left) // col_w)
            if 0 <= col < COL_COUNT and is_valid_location(self.board, col):
                if self.make_move(col):
                    self.trigger_ai_if_needed()
            return

        # Sidebar Buttons
        for btn in self.ui_buttons:
            if btn["rect"].collidepoint(pos):
                btn["action"]()
                return

    # --- Drawing Routines ---
    def draw(self):
        screen.fill(COLOR_BG)

        # Header Titles
        title_surf = FONT_TITLE.render("CONNECT 4 AI", True, COLOR_TEXT_MAIN)
        screen.blit(title_surf, (32, 28))

        sub_surf = FONT_SUBTITLE.render("Real-time Evaluation • Deep Q-Network • Minimax Tree Search", True, COLOR_TEXT_MUTED)
        screen.blit(sub_surf, (32, 60))

        # 1. Win Rate Evaluation Bar
        self.draw_eval_bar()

        # 2. Main Game Board
        self.draw_board()

        # 3. Sidebar Controls
        self.draw_sidebar()

    def draw_eval_bar(self):
        # Outer Card Container
        pygame.draw.rect(screen, COLOR_CARD, RECT_EVAL_BAR, border_radius=12)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, RECT_EVAL_BAR, width=2, border_radius=12)

        inner_rect = RECT_EVAL_BAR.inflate(-6, -6)
        p1_h = int(inner_rect.height * self.display_p1_rate)
        p2_h = inner_rect.height - p1_h

        # Top is Red (P1), Bottom is Yellow (P2)
        red_rect = pygame.Rect(inner_rect.left, inner_rect.top, inner_rect.width, p1_h)
        yellow_rect = pygame.Rect(inner_rect.left, inner_rect.top + p1_h, inner_rect.width, p2_h)

        pygame.draw.rect(screen, COLOR_RED, red_rect, border_top_left_radius=8, border_top_right_radius=8)
        pygame.draw.rect(screen, COLOR_YELLOW, yellow_rect, border_bottom_left_radius=8, border_bottom_right_radius=8)

        # Divider line
        div_y = inner_rect.top + p1_h
        pygame.draw.line(screen, (255, 255, 255), (inner_rect.left, div_y), (inner_rect.right, div_y), 2)

        # Percentage Numbers
        p1_pct = int(round(self.display_p1_rate * 100))
        p2_pct = 100 - p1_pct

        txt_p1 = FONT_SCORE.render(f"{p1_pct}%", True, (255, 255, 255))
        screen.blit(txt_p1, (RECT_EVAL_BAR.centerx - txt_p1.get_width() // 2, RECT_EVAL_BAR.top + 8))

        txt_p2 = FONT_SCORE.render(f"{p2_pct}%", True, (255, 255, 255))
        screen.blit(txt_p2, (RECT_EVAL_BAR.centerx - txt_p2.get_width() // 2, RECT_EVAL_BAR.bottom - 24))

        # Bar Labels
        label_surf = FONT_SMALL.render("EVAL", True, COLOR_TEXT_MUTED)
        screen.blit(label_surf, (RECT_EVAL_BAR.centerx - label_surf.get_width() // 2, RECT_EVAL_BAR.top - 20))

        src_surf = FONT_SMALL.render(self.eval_source, True, COLOR_ACCENT)
        screen.blit(src_surf, (RECT_EVAL_BAR.centerx - src_surf.get_width() // 2, RECT_EVAL_BAR.bottom + 8))

    def draw_board(self):
        # Outer Frame
        pygame.draw.rect(screen, COLOR_BOARD_FRAME, RECT_BOARD_AREA, border_radius=18)
        inner_board = RECT_BOARD_AREA.inflate(-12, -12)
        pygame.draw.rect(screen, COLOR_BOARD_INNER, inner_board, border_radius=14)

        col_w = inner_board.width / COL_COUNT
        row_h = inner_board.height / ROW_COUNT
        radius = min(col_w, row_h) * 0.40

        # Draw Hover Column Beam
        if self.hover_col is not None and not self.anim and not self.game_over:
            beam_x = inner_board.left + self.hover_col * col_w
            beam_surf = pygame.Surface((int(col_w), int(inner_board.height)), pygame.SRCALPHA)
            beam_color = (255, 255, 255, 18)
            beam_surf.fill(beam_color)
            screen.blit(beam_surf, (beam_x, inner_board.top))

        # Draw Slots & Placed Discs with Anti-Aliasing
        for c in range(COL_COUNT):
            for r in range(ROW_COUNT):
                cx = inner_board.left + (c + 0.5) * col_w
                cy = inner_board.bottom - (r + 0.5) * row_h
                piece = self.board[r][c]

                # Don't draw static piece if currently dropping here
                if self.anim and self.anim.col == c and self.anim.target_row == r:
                    piece_to_draw = 0
                else:
                    piece_to_draw = piece

                # Empty dark cavity
                draw_aa_circle(screen, cx, cy, radius, COLOR_SLOT_EMPTY)

                # Render Disc
                if piece_to_draw == PLAYER_PIECE:
                    self.draw_piece_circle(cx, cy, radius, COLOR_RED, COLOR_RED_HIGHLIGHT)
                elif piece_to_draw == AI_PIECE:
                    self.draw_piece_circle(cx, cy, radius, COLOR_YELLOW, COLOR_YELLOW_HIGHLIGHT)

        # Draw Falling Piece
        if self.anim:
            cx = inner_board.left + (self.anim.col + 0.5) * col_w
            cy = self.anim.current_y
            color = COLOR_RED if self.anim.piece == PLAYER_PIECE else COLOR_YELLOW
            highlight = COLOR_RED_HIGHLIGHT if self.anim.piece == PLAYER_PIECE else COLOR_YELLOW_HIGHLIGHT
            self.draw_piece_circle(cx, cy, radius, color, highlight)

        # Draw Hover Disc Preview (top indicator)
        if self.hover_col is not None and not self.anim and not self.game_over:
            hx = inner_board.left + (self.hover_col + 0.5) * col_w
            hy = RECT_BOARD_AREA.top - 18
            color = COLOR_RED if self.turn == 0 else COLOR_YELLOW
            hover_surf = pygame.Surface((int(radius * 2 + 4), int(radius * 2 + 4)), pygame.SRCALPHA)
            center = int(radius + 2)
            pygame.gfxdraw.filled_circle(hover_surf, center, center, int(radius * 0.85), (*color, 160))
            pygame.gfxdraw.aacircle(hover_surf, center, center, int(radius * 0.85), (255, 255, 255, 220))
            screen.blit(hover_surf, (hx - radius - 2, hy - radius - 2))

        # Draw 4-in-a-row Winning Glow Highlight
        if self.win_coords:
            pulse = (math.sin(time.time() * 8) + 1) * 0.5
            glow_radius = radius + 3 + pulse * 4
            points = []
            for r, c in self.win_coords:
                wx = inner_board.left + (c + 0.5) * col_w
                wy = inner_board.bottom - (r + 0.5) * row_h
                points.append((wx, wy))
                pygame.draw.circle(screen, COLOR_WIN_GLOW, (int(wx), int(wy)), int(glow_radius), 3)

            if len(points) == 4:
                pygame.draw.line(screen, COLOR_WIN_GLOW, points[0], points[3], 5)

    def draw_piece_circle(self, cx, cy, radius, base_color, highlight_color):
        """Draws a clean, anti-aliased 3D-styled token."""
        draw_aa_circle(screen, cx, cy, radius, base_color)
        # Inner glossy ring highlight
        hx = cx - radius * 0.22
        hy = cy - radius * 0.22
        draw_aa_circle(screen, hx, hy, radius * 0.45, highlight_color)
        # Center filler to create specular shine effect
        draw_aa_circle(screen, hx + 1, hy + 1, radius * 0.35, base_color)
        # Subtle perimeter rim
        pygame.gfxdraw.aacircle(screen, int(round(cx)), int(round(cy)), int(round(radius)), (0, 0, 0, 70))

    def draw_sidebar(self):
        # Card Background
        pygame.draw.rect(screen, COLOR_CARD, RECT_SIDEBAR, border_radius=16)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, RECT_SIDEBAR, width=2, border_radius=16)

        self.ui_buttons = []
        y = RECT_SIDEBAR.top + 20
        pad_x = RECT_SIDEBAR.left + 20
        content_w = RECT_SIDEBAR.width - 40

        # --- Section 1: Match Status Header ---
        self.draw_status_card(pad_x, y, content_w)
        y += 122

        # --- Section 2: Opponent Mode ---
        txt_op = FONT_BOLD.render("Opponent Type", True, COLOR_TEXT_MAIN)
        screen.blit(txt_op, (pad_x, y))
        y += 28

        opponents = [
            ("DQN", "🤖 DQN Neural Network"),
            ("Minimax", "🧠 Minimax Tree Search"),
            ("Random", "🎲 Tactical Random AI"),
            ("Human", "👤 Local 2-Player (PvP)")
        ]

        for key, label in opponents:
            is_active = (self.opponent_type == key)
            btn_rect = pygame.Rect(pad_x, y, content_w, 36)
            self.draw_radio_item(btn_rect, label, is_active, lambda k=key: self.set_opponent(k))
            y += 42

        # Minimax Depth Selector
        if self.opponent_type == "Minimax":
            depth_label = FONT_SMALL.render("Depth:", True, COLOR_TEXT_MUTED)
            screen.blit(depth_label, (pad_x + 6, y + 3))
            depths = [(2, "Easy 2"), (4, "Med 4"), (5, "Hard 5")]
            sub_w = 64
            for i, (d, d_name) in enumerate(depths):
                sub_rect = pygame.Rect(pad_x + 60 + i * (sub_w + 6), y - 2, sub_w, 28)
                is_d_active = (self.minimax_depth == d)
                self.draw_pill_button(sub_rect, d_name, is_d_active, lambda depth=d: self.set_depth(depth))
            y += 36

        y += 10

        # --- Section 3: First Move (Turn Order) ---
        txt_order = FONT_BOLD.render("First Move (Red Disc)", True, COLOR_TEXT_MAIN)
        screen.blit(txt_order, (pad_x, y))
        y += 28

        orders = [
            ("Human", "🔴 Player First"),
            ("AI", "🟡 AI First")
        ]
        half_w = (content_w - 10) // 2
        for i, (key, label) in enumerate(orders):
            rect = pygame.Rect(pad_x + i * (half_w + 10), y, half_w, 34)
            is_active = (self.first_mover == key)
            self.draw_pill_button(rect, label, is_active, lambda k=key: self.set_first_mover(k))
        y += 52

        # --- Section 4: Game Actions ---
        btn_new = pygame.Rect(pad_x, y, content_w, 42)
        self.draw_action_button(btn_new, "✨ New Game", COLOR_BTN_ACTIVE, self.reset_game)
        y += 50

        btn_undo = pygame.Rect(pad_x, y, (content_w - 10) // 2, 38)
        self.draw_action_button(btn_undo, "↩ Undo Move", COLOR_BTN_DEFAULT, self.undo_move)

        btn_ai_mode = pygame.Rect(pad_x + (content_w - 10) // 2 + 10, y, (content_w - 10) // 2, 38)
        ai_vs_ai_color = COLOR_BTN_DANGER if self.ai_vs_ai else COLOR_BTN_DEFAULT
        ai_vs_ai_txt = "⏹ Stop Watch" if self.ai_vs_ai else "🤖 AI vs AI"
        self.draw_action_button(btn_ai_mode, ai_vs_ai_txt, ai_vs_ai_color, self.toggle_ai_vs_ai)

    def draw_status_card(self, x, y, w):
        rect = pygame.Rect(x, y, w, 106)
        pygame.draw.rect(screen, (15, 23, 42), rect, border_radius=12)
        pygame.draw.rect(screen, COLOR_CARD_BORDER, rect, width=1, border_radius=12)

        # Status text rendering
        if self.game_over:
            if self.winner == PLAYER_PIECE:
                status_title = "🎉 Red Player Wins!"
                status_color = COLOR_RED
            elif self.winner == AI_PIECE:
                status_title = "🏆 Yellow Player Wins!"
                status_color = COLOR_YELLOW
            else:
                status_title = "🤝 Game Drawn!"
                status_color = COLOR_TEXT_MUTED
            sub_info = "Click [New Game] to restart"
        elif self.ai_worker.is_thinking:
            dots = "." * (int(time.time() * 3) % 4)
            status_title = f"AI Thinking{dots}"
            status_color = COLOR_ACCENT
            sub_info = f"Engine: {self.get_opponent_display_name()}"
        else:
            if self.turn == 0:
                status_title = "🔴 Turn: Red (P1)"
                status_color = COLOR_RED
                who = "Player" if (self.first_mover == "Human" and not self.ai_vs_ai) else "AI Agent"
            else:
                status_title = "🟡 Turn: Yellow (P2)"
                status_color = COLOR_YELLOW
                who = "Player" if (self.first_mover == "AI" and not self.ai_vs_ai) else "AI Agent"
            sub_info = f"Controlled by {who} • Ply: {len(self.move_history)}"

        txt_surf = FONT_BOLD.render(status_title, True, status_color)
        screen.blit(txt_surf, (x + 14, y + 14))

        sub_surf = FONT_SMALL.render(sub_info, True, COLOR_TEXT_MUTED)
        screen.blit(sub_surf, (x + 14, y + 42))

        # Mode Badge
        badge_rect = pygame.Rect(x + 14, y + 70, w - 28, 24)
        pygame.draw.rect(screen, (30, 41, 59), badge_rect, border_radius=6)
        opp_name = self.get_opponent_display_name()
        badge_surf = FONT_SMALL.render(f"Mode: {opp_name}", True, COLOR_ACCENT)
        screen.blit(badge_surf, (x + 22, y + 74))

    def get_opponent_display_name(self):
        if self.ai_vs_ai:
            return "AI vs AI Spectator"
        if self.opponent_type == "DQN":
            return "DQN Deep Neural Net"
        elif self.opponent_type == "Minimax":
            depth_map = {2: "Easy", 4: "Med", 5: "Hard"}
            return f"Minimax ({depth_map.get(self.minimax_depth, '')} - Depth {self.minimax_depth})"
        elif self.opponent_type == "Random":
            return "Tactical Random AI"
        return "Pass & Play (2P)"

    def draw_radio_item(self, rect, label, is_active, on_click):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)

        bg_color = (37, 99, 235, 40) if is_active else ((51, 65, 85) if hover else (15, 23, 42))
        border_color = COLOR_ACCENT if is_active else ((100, 116, 139) if hover else COLOR_CARD_BORDER)

        pygame.draw.rect(screen, bg_color[:3], rect, border_radius=8)
        pygame.draw.rect(screen, border_color, rect, width=2 if is_active else 1, border_radius=8)

        # Smooth Radio Disc
        dot_center = (rect.left + 20, rect.centery)
        pygame.gfxdraw.aacircle(screen, dot_center[0], dot_center[1], 7, border_color)
        if is_active:
            draw_aa_circle(screen, dot_center[0], dot_center[1], 4, COLOR_ACCENT)

        txt_surf = FONT_NORMAL.render(label, True, COLOR_TEXT_MAIN if is_active else COLOR_TEXT_MUTED)
        screen.blit(txt_surf, (rect.left + 36, rect.centery - txt_surf.get_height() // 2))

        self.ui_buttons.append({"rect": rect, "action": on_click})

    def draw_pill_button(self, rect, label, is_active, on_click):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)

        bg = COLOR_BTN_ACTIVE if is_active else (COLOR_BTN_HOVER if hover else (15, 23, 42))
        border = COLOR_ACCENT if is_active else COLOR_CARD_BORDER

        pygame.draw.rect(screen, bg, rect, border_radius=6)
        pygame.draw.rect(screen, border, rect, width=1, border_radius=6)

        txt = FONT_SMALL.render(label, True, (255, 255, 255) if is_active else COLOR_TEXT_MUTED)
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        self.ui_buttons.append({"rect": rect, "action": on_click})

    def draw_action_button(self, rect, label, color, on_click):
        mouse_pos = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse_pos)

        bg = [min(255, c + 20) for c in color] if hover else color
        pygame.draw.rect(screen, bg, rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255, 45), rect, width=1, border_radius=8)

        txt = FONT_BOLD.render(label, True, (255, 255, 255))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        self.ui_buttons.append({"rect": rect, "action": on_click})

    # --- Configuration Setters ---
    def set_opponent(self, opp):
        if self.opponent_type != opp:
            self.opponent_type = opp
            self.ai_vs_ai = False
            self.reset_game()

    def set_depth(self, depth):
        if self.minimax_depth != depth:
            self.minimax_depth = depth
            if self.opponent_type == "Minimax":
                self.reset_game()

    def set_first_mover(self, mover):
        if self.first_mover != mover:
            self.first_mover = mover
            self.reset_game()

    def toggle_ai_vs_ai(self):
        self.ai_vs_ai = not self.ai_vs_ai
        self.reset_game()


if __name__ == "__main__":
    app = Connect4GUI()
    app.run()