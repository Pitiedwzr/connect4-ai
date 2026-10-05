"""Frozen, reproducible evaluation without exploration or training opponents."""
import argparse
import json
import math
import random

import torch

from agent import DuelingConnect4Net
from game import (create_board, drop_piece, get_candidate_moves, get_next_open_row,
                  get_state_tensor, get_valid_locations, minimax, winning_move)


def network_move(net, board, my_piece, opp_piece, raw=False):
    candidates = (get_valid_locations(board) if raw else
                  get_candidate_moves(board, my_piece, opp_piece))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    device = next(net.parameters()).device
    with torch.no_grad():
        q = net(get_state_tensor(board, my_piece, opp_piece).to(device))[0].cpu().tolist()
    return max(candidates, key=lambda c: q[c])


def teacher_move(board, my_piece, opp_piece, depth):
    """Search only actions admitted by the same tactical policy as the DQN."""
    candidates = get_candidate_moves(board, my_piece, opp_piece)
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    best, best_score = candidates[0], -math.inf
    for col in candidates:
        after = board.copy()
        drop_piece(after, get_next_open_row(after, col), col, my_piece)
        _, score = minimax(after, depth - 1, -math.inf, math.inf, False,
                           ai_piece=my_piece, player_piece=opp_piece)
        if score > best_score:
            best, best_score = col, score
    return best


def evaluate_network(net, games=20, seed=12345, depths=(2, 4), raw=False, opening_plies=4,
                     move_fn=None, extra_opponents=None):
    """Report W/D/L by seat with paired, seeded opening positions.

    The first game starts empty. Other games use random legal opening prefixes
    to avoid repeating one deterministic minimax matchup. Preserve training RNG
    and model mode. 'First'/'second' describe red/yellow seat, including prefixes.
    """
    if games < 1:
        raise ValueError("games must be positive (games per seat per opponent)")
    if not 0 <= opening_plies <= 12 or any(depth < 1 for depth in depths):
        raise ValueError("opening_plies must be in [0,12] and depths must be positive")
    if net is None and move_fn is None:
        raise ValueError("A network or move function is required")
    rng_state, was_training = random.getstate(), net.training if net is not None else None
    if net is not None:
        net.eval()
    results = {}
    extra_opponents = extra_opponents or {}
    try:
        opponents = (["random", "tactical_random"] + [f"minimax_{d}" for d in depths]
                     + list(extra_opponents))
        for opponent_index, opponent in enumerate(opponents):
            results[opponent] = {}
            for seat in (1, 2):
                counts = dict(wins=0, draws=0, losses=0)
                for game_index in range(games):
                    random.seed(seed + opponent_index * 100000 + game_index)
                    board, turn = create_board(), 1
                    # Both seats see the same prefix. Twelve plies cannot fill
                    # the board; discard any prefix with an early win.
                    if game_index and opening_plies:
                        while True:
                            board, turn, terminal = create_board(), 1, False
                            for _ in range(random.randint(1, opening_plies)):
                                col = random.choice(get_valid_locations(board))
                                drop_piece(board, get_next_open_row(board, col), col, turn)
                                if winning_move(board, turn):
                                    terminal = True
                                    break
                                turn = 3 - turn
                            if not terminal:
                                break
                    while True:
                        legal = get_valid_locations(board)
                        if not legal:
                            counts["draws"] += 1
                            break
                        if turn == seat:
                            col = (move_fn(board, seat, 3 - seat) if move_fn is not None else
                                   network_move(net, board, seat, 3 - seat, raw=raw))
                        elif opponent in extra_opponents:
                            col = extra_opponents[opponent](board, turn, 3 - turn)
                        elif opponent == "random":
                            col = random.choice(legal)
                        elif opponent == "tactical_random":
                            col = random.choice(get_candidate_moves(board, turn, 3 - turn))
                        else:
                            col, _ = minimax(board, int(opponent.split("_")[1]),
                                             -math.inf, math.inf, True,
                                             ai_piece=turn, player_piece=3 - turn)
                        if col not in legal:
                            raise RuntimeError(f"Illegal move {col} from {opponent}")
                        drop_piece(board, get_next_open_row(board, col), col, turn)
                        if winning_move(board, turn):
                            counts["wins" if turn == seat else "losses"] += 1
                            break
                        turn = 3 - turn
                results[opponent]["first" if seat == 1 else "second"] = counts
    finally:
        random.setstate(rng_state)
        if net is not None:
            net.train(was_training)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("dqn", "alphazero", "equinox"), default="dqn")
    parser.add_argument("--model", help="Checkpoint path; defaults to the selected agent's checkpoint")
    parser.add_argument("--games", type=int, default=20, help="Games per seat per opponent")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--depths", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--raw", action="store_true", help="Measure the network without tactical assistance")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--opening-plies", type=int, default=4, help="Maximum seeded opening prefix; 0 uses empty board only")
    parser.add_argument("--simulations", type=int, default=128, help="AlphaZero search budget per move")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--search-policy", choices=("puct", "gumbel"), help="Equinox only; defaults to checkpoint metadata")
    parser.add_argument("--dqn-opponent", help="Optional DQN checkpoint to include as an opponent")
    args = parser.parse_args()
    if any(d < 1 for d in args.depths):
        parser.error("depths must be positive")
    if args.games < 1 or not 0 <= args.opening_plies <= 12:
        parser.error("games must be positive and opening plies in [0,12]")
    if args.simulations < 1 or args.c_puct <= 0:
        parser.error("simulations and c-puct must be positive")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu":
        torch.set_num_threads(1)
    move_fn = None
    if args.agent == "equinox":
        from connect4_jax.agent import AlphaZeroAgent
        from connect4_jax.checkpoint import DEFAULT_MODEL_PATH
        import numpy as np
        player = AlphaZeroAgent.from_checkpoint(args.model or DEFAULT_MODEL_PATH, args.simulations, args.c_puct, args.search_policy)
        if (player.model.config.rows, player.model.config.cols, player.model.config.connect) != (6, 7, 4):
            parser.error("The baseline benchmark requires a standard 6x7 Connect 4 checkpoint")
        net = None
        move_fn = lambda board, my, other: player.get_move(board, my, other, raw=args.raw)
        logits, value = player.predict(np.zeros((6, 7), np.int8), 1)
        probabilities = np.exp(logits - logits.max())
        opening_policy = (probabilities / probabilities.sum()).tolist()
        report = dict(agent="equinox", mode="raw" if args.raw else "mctx", device="cpu",
                      search_policy=player.settings.policy,
                      simulations=0 if args.raw else args.simulations, c_puct=args.c_puct,
                      opening_policy=opening_policy, opening_value=value,
                      raw_opening_column=int(np.argmax(logits)))
    elif args.agent == "alphazero":
        from alphazero import AlphaZeroAgent, DEFAULT_MODEL_PATH, Position, load_checkpoint
        net, _ = load_checkpoint(args.model or DEFAULT_MODEL_PATH, device)
        if (net.config.rows, net.config.cols, net.config.connect) != (6, 7, 4):
            parser.error("The baseline benchmark currently uses standard 6x7 Connect 4 positions")
        player = AlphaZeroAgent(net, args.simulations, args.c_puct)
        move_fn = lambda board, my, other: player.get_move(board, my, other, raw=args.raw)
        with torch.no_grad():
            logits, value = net(torch.from_numpy(Position.empty(net.config).encode()).unsqueeze(0).to(device))
            opening_policy = torch.softmax(logits[0], dim=0).cpu().tolist()
        report = dict(agent="alphazero", mode="raw" if args.raw else "mcts",
                      simulations=0 if args.raw else args.simulations, c_puct=args.c_puct,
                      opening_policy=opening_policy, opening_value=value.item(),
                      raw_opening_column=max(range(7), key=lambda c: opening_policy[c]))
    else:
        net = DuelingConnect4Net().to(device)
        net.load_state_dict(torch.load(args.model or "model/connect4_model_selfplay.pth",
                                       map_location=device, weights_only=True))
        net.eval()
        with torch.no_grad():
            opening_q = net(get_state_tensor(create_board(), 1, 2).to(device))[0].cpu().tolist()
        report = dict(agent="dqn", mode="raw" if args.raw else "tactical", opening_q=opening_q,
                      raw_opening_column=max(range(7), key=lambda c: opening_q[c]))
    extra_opponents = {}
    if args.dqn_opponent:
        opponent = DuelingConnect4Net().to(device)
        opponent.load_state_dict(torch.load(args.dqn_opponent, map_location=device, weights_only=True))
        opponent.eval()
        extra_opponents["dqn"] = lambda board, my, other: network_move(opponent, board, my, other)
    report.update(seed=args.seed, opening_plies=args.opening_plies,
                  results=evaluate_network(net, args.games, args.seed, tuple(args.depths), args.raw,
                                           args.opening_plies, move_fn, extra_opponents))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
