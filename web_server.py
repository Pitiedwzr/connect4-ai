"""
FastAPI Local Web Server for Connect 4 AlphaGo-Style GUI & Analysis Studio.
Provides real-time game state, MCTS evaluation streams, move reviews,
and sandbox branch exploration.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from game import (
    COL_COUNT,
    ROW_COUNT,
    create_board,
    drop_piece,
    get_candidate_moves,
    get_next_open_row,
    get_valid_locations,
    is_terminal_node,
    is_valid_location,
    winning_move,
)
from analysis_engine import (
    AnalysisResult,
    CandidateMove,
    Connect4AnalysisEngine,
    classify_move_quality,
)

app = FastAPI(title="Connect 4 AlphaGo Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)
analysis_engine = Connect4AnalysisEngine(default_simulations=128)


@dataclass
class MoveRecord:
    ply: int
    row: int
    col: int
    piece: int
    timestamp: float
    win_rate_red: float
    win_rate_yellow: float
    quality_label: str = ""
    quality_color: str = ""
    delta_win_rate: float = 0.0


class GameSession:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.int8)
        self.to_play = 1  # 1 = Red, 2 = Yellow
        self.game_over = False
        self.winner = 0   # 0 = Ongoing/Draw, 1 = Red, 2 = Yellow
        self.win_coords: Optional[List[List[int]]] = None
        self.history: List[MoveRecord] = []
        self.mode = "human_vs_ai"  # "human_vs_ai", "ai_vs_ai", "human_vs_human", "analysis"
        self.player_red = "human"     # "human", "alphazero", "dqn", "minimax"
        self.player_yellow = "alphazero"
        self.simulations = 128
        self.current_analysis: Optional[AnalysisResult] = None
        self.is_thinking = False

        # Sandbox state
        self.sandbox_active = False
        self.sandbox_board: Optional[np.ndarray] = None
        self.sandbox_to_play = 1
        self.sandbox_history: List[int] = []

    def reset(self, player_red="human", player_yellow="alphazero", simulations=128):
        self.board = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.int8)
        self.to_play = 1
        self.game_over = False
        self.winner = 0
        self.win_coords = None
        self.history = []
        self.player_red = player_red
        self.player_yellow = player_yellow
        self.simulations = simulations
        self.current_analysis = None
        self.is_thinking = False
        self.sandbox_active = False
        self.sandbox_board = None

    def get_effective_board(self) -> np.ndarray:
        if self.sandbox_active and self.sandbox_board is not None:
            return self.sandbox_board
        return self.board

    def get_effective_to_play(self) -> int:
        if self.sandbox_active:
            return self.sandbox_to_play
        return self.to_play


game_session = GameSession()


# --- Request Models ---
class MoveRequest(BaseModel):
    col: int
    auto_ai: bool = False


class ResetRequest(BaseModel):
    player_red: str = "human"
    player_yellow: str = "alphazero"
    simulations: int = 128


class AnalyzeRequest(BaseModel):
    engine_type: str = "alphazero"
    simulations: int = 256
    custom_board: Optional[List[List[int]]] = None
    to_play: Optional[int] = None


class JumpRequest(BaseModel):
    ply: int


class SandboxToggleRequest(BaseModel):
    active: bool


# --- API Routes ---

@app.get("/api/state")
async def get_state():
    board = game_session.get_effective_board()
    to_play = game_session.get_effective_to_play()

    # If analysis is not yet computed for current position, run quick analysis
    if game_session.current_analysis is None:
        loop = asyncio.get_event_loop()
        game_session.current_analysis = await loop.run_in_executor(
            executor,
            analysis_engine.analyze,
            board.copy(),
            to_play,
            "alphazero",
            game_session.simulations,
        )

    return {
        "board": board.tolist(),
        "to_play": to_play,
        "game_over": game_session.game_over,
        "winner": game_session.winner,
        "win_coords": game_session.win_coords,
        "history": [asdict(m) for m in game_session.history],
        "player_red": game_session.player_red,
        "player_yellow": game_session.player_yellow,
        "simulations": game_session.simulations,
        "analysis": game_session.current_analysis.to_dict() if game_session.current_analysis else None,
        "is_thinking": game_session.is_thinking,
        "sandbox_active": game_session.sandbox_active,
    }


@app.post("/api/move")
async def make_move(req: MoveRequest):
    if game_session.game_over and not game_session.sandbox_active:
        raise HTTPException(status_code=400, detail="Game is already over")

    board = game_session.get_effective_board()
    to_play = game_session.get_effective_to_play()
    col = req.col

    if not is_valid_location(board, col):
        raise HTTPException(status_code=400, detail=f"Invalid move in column {col}")

    # Prior analysis for blunder classification
    prior_analysis = game_session.current_analysis

    # Place piece
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, to_play)

    # Check terminal state
    is_win = winning_move(board, to_play)
    is_draw = is_terminal_node(board) and not is_win

    # Move quality evaluation
    quality_label = ""
    quality_color = ""
    delta_win_rate = 0.0

    if prior_analysis and prior_analysis.candidates:
        best_cand = prior_analysis.candidates[0]
        chosen_cand = next((c for c in prior_analysis.candidates if c.col == col), None)
        if chosen_cand:
            delta_win_rate = round(max(0.0, best_cand.win_rate - chosen_cand.win_rate), 1)
        else:
            delta_win_rate = round(best_cand.win_rate, 1)
        quality_label, quality_color = classify_move_quality(delta_win_rate)

    last_ai_move = None

    if not game_session.sandbox_active:
        from game import get_winning_coordinates
        if is_win:
            game_session.game_over = True
            game_session.winner = to_play
            game_session.win_coords = [list(c) for c in get_winning_coordinates(board, to_play)]
        elif is_draw:
            game_session.game_over = True
            game_session.winner = 0

        game_session.history.append(
            MoveRecord(
                ply=len(game_session.history) + 1,
                row=row,
                col=col,
                piece=to_play,
                timestamp=time.time(),
                win_rate_red=prior_analysis.win_rate_red if prior_analysis else 50.0,
                win_rate_yellow=prior_analysis.win_rate_yellow if prior_analysis else 50.0,
                quality_label=quality_label,
                quality_color=quality_color,
                delta_win_rate=delta_win_rate,
            )
        )
        game_session.to_play = 3 - to_play
        next_to_play = game_session.to_play

        # Auto AI response in a single smooth cycle
        if not game_session.game_over and req.auto_ai:
            next_player = game_session.player_red if next_to_play == 1 else game_session.player_yellow
            if next_player != "human":
                loop = asyncio.get_event_loop()
                ai_analysis = await loop.run_in_executor(
                    executor,
                    analysis_engine.analyze,
                    board.copy(),
                    next_to_play,
                    next_player if next_player in ("alphazero", "dqn") else "alphazero",
                    game_session.simulations,
                )
                ai_col = ai_analysis.best_move
                if ai_col is None:
                    valid_cols = get_valid_locations(board)
                    ai_col = valid_cols[0] if valid_cols else None

                if ai_col is not None:
                    ai_row = get_next_open_row(board, ai_col)
                    drop_piece(board, ai_row, ai_col, next_to_play)
                    ai_win = winning_move(board, next_to_play)
                    ai_draw = is_terminal_node(board) and not ai_win

                    if ai_win:
                        game_session.game_over = True
                        game_session.winner = next_to_play
                        game_session.win_coords = [list(c) for c in get_winning_coordinates(board, next_to_play)]
                    elif ai_draw:
                        game_session.game_over = True
                        game_session.winner = 0

                    game_session.history.append(
                        MoveRecord(
                            ply=len(game_session.history) + 1,
                            row=ai_row,
                            col=ai_col,
                            piece=next_to_play,
                            timestamp=time.time(),
                            win_rate_red=ai_analysis.win_rate_red,
                            win_rate_yellow=ai_analysis.win_rate_yellow,
                            quality_label="Best Move",
                            quality_color="#10B981",
                            delta_win_rate=0.0,
                        )
                    )
                    last_ai_move = {"row": ai_row, "col": ai_col, "piece": next_to_play}
                    game_session.to_play = 3 - next_to_play
                    next_to_play = game_session.to_play
    else:
        game_session.sandbox_history.append(col)
        game_session.sandbox_to_play = 3 - to_play
        next_to_play = game_session.sandbox_to_play

    # Compute updated analysis for the new position (human's upcoming turn)
    loop = asyncio.get_event_loop()
    game_session.current_analysis = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        board.copy(),
        next_to_play,
        "alphazero",
        game_session.simulations,
    )

    state = await get_state()
    state["last_ai_move"] = last_ai_move
    return state


@app.post("/api/ai_move")
async def trigger_ai_move():
    if game_session.game_over:
        return await get_state()

    to_play = game_session.get_effective_to_play()
    current_player = game_session.player_red if to_play == 1 else game_session.player_yellow

    if current_player == "human":
        raise HTTPException(status_code=400, detail="Current turn is human player")

    game_session.is_thinking = True
    try:
        board = game_session.get_effective_board()
        loop = asyncio.get_event_loop()

        # Run analysis once to find best move and candidates
        analysis = await loop.run_in_executor(
            executor,
            analysis_engine.analyze,
            board.copy(),
            to_play,
            current_player if current_player in ("alphazero", "dqn") else "alphazero",
            game_session.simulations,
        )

        best_col = analysis.best_move
        if best_col is None:
            valid_cols = get_valid_locations(board)
            best_col = valid_cols[0] if valid_cols else None

        if best_col is None:
            return await get_state()

        row = get_next_open_row(board, best_col)
        drop_piece(board, row, best_col, to_play)
        is_win = winning_move(board, to_play)
        is_draw = is_terminal_node(board) and not is_win

        from game import get_winning_coordinates
        if is_win:
            game_session.game_over = True
            game_session.winner = to_play
            game_session.win_coords = [list(c) for c in get_winning_coordinates(board, to_play)]
        elif is_draw:
            game_session.game_over = True
            game_session.winner = 0

        game_session.history.append(
            MoveRecord(
                ply=len(game_session.history) + 1,
                row=row,
                col=best_col,
                piece=to_play,
                timestamp=time.time(),
                win_rate_red=analysis.win_rate_red,
                win_rate_yellow=analysis.win_rate_yellow,
                quality_label="Best Move",
                quality_color="#10B981",
                delta_win_rate=0.0,
            )
        )
        game_session.to_play = 3 - to_play
        next_to_play = game_session.to_play

        # Compute analysis for human's next move
        game_session.current_analysis = await loop.run_in_executor(
            executor,
            analysis_engine.analyze,
            board.copy(),
            next_to_play,
            "alphazero",
            game_session.simulations,
        )

        state = await get_state()
        state["last_ai_move"] = {"row": row, "col": best_col, "piece": to_play}
        return state
    finally:
        game_session.is_thinking = False

    return await get_state()


@app.post("/api/analyze")
async def analyze_position(req: AnalyzeRequest):
    board = np.array(req.custom_board, dtype=np.int8) if req.custom_board else game_session.get_effective_board()
    to_play = req.to_play or game_session.get_effective_to_play()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        board.copy(),
        to_play,
        req.engine_type,
        req.simulations,
    )
    return result.to_dict()


@app.post("/api/undo")
async def undo_move():
    if not game_session.history:
        return await get_state()

    # Determine how many steps to undo
    is_pv_ai = (game_session.player_red == "human" and game_session.player_yellow != "human") or \
               (game_session.player_yellow == "human" and game_session.player_red != "human")

    steps = 2 if (is_pv_ai and len(game_session.history) >= 2 and not game_session.game_over) else 1

    for _ in range(steps):
        if game_session.history:
            last = game_session.history.pop()
            game_session.board[last.row, last.col] = 0
            game_session.to_play = last.piece

    game_session.game_over = False
    game_session.winner = 0
    game_session.win_coords = None

    loop = asyncio.get_event_loop()
    game_session.current_analysis = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        game_session.board.copy(),
        game_session.to_play,
        "alphazero",
        game_session.simulations,
    )
    return await get_state()


@app.post("/api/reset")
async def reset_game(req: ResetRequest):
    game_session.reset(
        player_red=req.player_red,
        player_yellow=req.player_yellow,
        simulations=req.simulations,
    )

    loop = asyncio.get_event_loop()
    game_session.current_analysis = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        game_session.board.copy(),
        1,
        "alphazero",
        req.simulations,
    )
    return await get_state()


@app.post("/api/sandbox")
async def toggle_sandbox(req: SandboxToggleRequest):
    game_session.sandbox_active = req.active
    if req.active:
        game_session.sandbox_board = game_session.board.copy()
        game_session.sandbox_to_play = game_session.to_play
        game_session.sandbox_history = []
    else:
        game_session.sandbox_board = None

    loop = asyncio.get_event_loop()
    game_session.current_analysis = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        game_session.get_effective_board().copy(),
        game_session.get_effective_to_play(),
        "alphazero",
        game_session.simulations,
    )
    return await get_state()


@app.post("/api/jump")
async def jump_to_ply(req: JumpRequest):
    """Reconstruct board at a specific historical ply for review."""
    target_ply = max(0, min(req.ply, len(game_session.history)))
    new_board = np.zeros((ROW_COUNT, COL_COUNT), dtype=np.int8)

    for i in range(target_ply):
        m = game_session.history[i]
        new_board[m.row, m.col] = m.piece

    to_play = 1 if target_ply % 2 == 0 else 2

    # Set as active sandbox exploration
    game_session.sandbox_active = True
    game_session.sandbox_board = new_board
    game_session.sandbox_to_play = to_play

    loop = asyncio.get_event_loop()
    game_session.current_analysis = await loop.run_in_executor(
        executor,
        analysis_engine.analyze,
        new_board.copy(),
        to_play,
        "alphazero",
        game_session.simulations,
    )
    return await get_state()


# Mount static directory for frontend
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>Connect 4 AlphaGo Studio</h2><p>Frontend static files loading...</p>")


def launch_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    import uvicorn
    if open_browser:
        threading_timer = asyncio.get_event_loop() if False else None
        # Open browser shortly after server startup
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    launch_server()
