"""Search controls shared by training, inference, and benchmark commands."""
import argparse


def add_search_arguments(parser, *, checkpoint_defaults=True, prefix=""):
    parser.add_argument(f"--{prefix}proven-win-priority", action=argparse.BooleanOptionalAction,
                        default=None if checkpoint_defaults else False,
                        help="Prioritize exact terminal wins already explored by MCTS; correct self-play targets too")
    parser.add_argument(f"--{prefix}prior-temperature", type=float,
                        default=None if checkpoint_defaults else 1.0,
                        help="Divide root and leaf policy logits by this positive value; >1 flattens priors")
    parser.add_argument(f"--{prefix}gumbel-q-scale", type=float,
                        default=None if checkpoint_defaults else 0.1,
                        help="Positive Gumbel Q-value scale (native default 0.1); ignored by PUCT")


def search_overrides(args, *, prefix=""):
    return {name: getattr(args, prefix + name)
            for name in ("proven_win_priority", "prior_temperature", "gumbel_q_scale")}
