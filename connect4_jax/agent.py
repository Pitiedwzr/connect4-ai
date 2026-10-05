"""CPU-only policy and mctx inference with the existing player interface."""
import threading

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import DEFAULT_MODEL_PATH, load_checkpoint
from .environment import encode, from_board, legal_actions
from .search import SearchConfig, search


@eqx.filter_jit
def _predict(model, state):
    return model(encode(state))


@eqx.filter_jit
def _search(model, state, settings):
    states = jax.tree.map(lambda x: x[None], state)
    result = search(model, states, jax.random.PRNGKey(0), settings, add_noise=False)
    summary = result.search_tree.summary()
    action = result.action[0] if settings.policy == "gumbel" else jnp.argmax(summary.visit_counts[0])
    return result.action_weights[0], summary.visit_counts[0], summary.value[0], action


class AlphaZeroAgent:
    def __init__(self, model, simulations=128, c_puct=1.5, search_policy="puct"):
        self.device = jax.devices("cpu")[0]
        self.model = jax.device_put(model, self.device)
        self.settings = SearchConfig(simulations=simulations, c_puct=c_puct, policy=search_policy)
        self.last_result = None
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path=DEFAULT_MODEL_PATH, simulations=128, c_puct=1.5, search_policy=None):
        model, metadata = load_checkpoint(path, jax.devices("cpu")[0])
        policy = search_policy or metadata.get("search", {}).get("policy", metadata.get("training_args", {}).get("search_policy", "puct"))
        return cls(model, simulations, c_puct, policy)

    def predict(self, board, to_play):
        with jax.default_device(self.device):
            state = from_board(board, to_play, self.model.config)
            logits, value = _predict(self.model, state)
            return np.asarray(logits), float(value)

    def warmup(self):
        board = np.zeros((self.model.config.rows, self.model.config.cols), np.int8)
        self.get_move(board, raw=True)
        self.get_move(board)

    def get_move(self, board, my_piece=1, opp_piece=2, raw=False):
        if my_piece not in (1, 2) or opp_piece != 3 - my_piece:
            raise ValueError("AlphaZero uses pieces 1 and 2")
        with self._lock, jax.default_device(self.device):
            state = from_board(board, my_piece, self.model.config)
            if bool(state.done):
                self.last_result = None
                return None
            legal = np.asarray(legal_actions(state))
            if raw:
                logits, _ = _predict(self.model, state)
                self.last_result = None
                return int(np.argmax(np.where(legal, np.asarray(logits), -np.inf)))
            policy, visits, value, action = _search(self.model, state, self.settings)
            action = int(action)
            if not legal[action]:
                raise RuntimeError("Search returned an illegal action")
            self.last_result = dict(policy=np.asarray(policy), visits=np.asarray(visits), value=float(value),
                                    action=action, search_policy=self.settings.policy)
            return action
