"""Play a policy/value checkpoint in the terminal, including larger boards."""
import argparse

import numpy as np
import torch

from alphazero import AlphaZeroAgent, DEFAULT_MODEL_PATH, Position, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ai-first", action="store_true")
    parser.add_argument("--raw", action="store_true", help="Use the policy without tree search")
    args = parser.parse_args()
    if args.simulations < 1:
        parser.error("simulations must be positive")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, _ = load_checkpoint(args.model, device)
    agent = AlphaZeroAgent(model, args.simulations)
    position = Position.empty(model.config)
    human = 2 if args.ai_first else 1
    while position.terminal_value() is None:
        print(np.flip(position.board(), axis=0))
        print("Columns:", list(range(position.config.cols)))
        if position.to_play == human:
            try:
                action = int(input("Your column (Ctrl+C to quit): "))
            except ValueError:
                print("Enter a column number.")
                continue
            if action not in position.legal_moves():
                print("That column is full or outside the board.")
                continue
        else:
            action = agent.get_move(position.board(), position.to_play, 3 - position.to_play, raw=args.raw)
            print(f"AlphaZero selected {action}")
        position = position.play(action)
    print(np.flip(position.board(), axis=0))
    print("Draw" if not position.winner else "You win" if position.winner == human else "AlphaZero wins")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGame ended.")
