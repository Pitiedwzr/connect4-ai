"""Play a portable Equinox checkpoint on CPU, including larger boards."""
import argparse

import jax
import numpy as np

from connect4_jax.agent import AlphaZeroAgent
from connect4_jax.checkpoint import DEFAULT_MODEL_PATH
from connect4_jax.environment import empty, legal_actions, step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--search-policy", choices=("puct", "gumbel"), help="Defaults to checkpoint metadata")
    parser.add_argument("--ai-first", action="store_true")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    agent = AlphaZeroAgent.from_checkpoint(args.model, args.simulations, search_policy=args.search_policy)
    config = agent.model.config
    human = 2 if args.ai_first else 1
    with jax.default_device(agent.device):
        position = empty(config)
        advance = jax.jit(lambda s, a: step(s, a, config))
        print("Warming up CPU inference...")
        if args.raw:
            agent.get_move(np.zeros((config.rows, config.cols)), raw=True)
        else:
            agent.warmup()
        while not bool(position.done):
            print(np.flip(np.asarray(position.board), axis=0))
            print("Columns:", list(range(config.cols)))
            if int(position.to_play) == human:
                try:
                    action = int(input("Your column (Ctrl+C to quit): "))
                except ValueError:
                    print("Enter a column number.")
                    continue
                if not 0 <= action < config.cols or not bool(legal_actions(position)[action]):
                    print("That column is full or outside the board.")
                    continue
            else:
                piece = int(position.to_play)
                action = agent.get_move(np.asarray(position.board), piece, 3 - piece, raw=args.raw)
                print(f"AlphaZero selected {action}")
            position = advance(position, action)
        print(np.flip(np.asarray(position.board), axis=0))
        winner = int(position.winner)
        print("Draw" if not winner else "You win" if winner == human else "AlphaZero wins")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGame ended.")
