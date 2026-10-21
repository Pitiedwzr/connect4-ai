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
def _search(model, state, settings, diagnostics=False):
    states = jax.tree.map(lambda x: x[None], state)
    result = search(model, states, jax.random.PRNGKey(0), settings, add_noise=False)
    summary = result.search_tree.summary()
    action = result.action[0] if settings.policy == "gumbel" else jnp.argmax(result.action_weights[0])
    output = (result.action_weights[0], summary.visit_counts[0], summary.value[0], action)
    if not diagnostics:
        return output
    tree = result.search_tree
    root = tree.ROOT_INDEX
    return (*output, dict(priors=jax.nn.softmax(tree.children_prior_logits[0, root]),
                         qvalues=summary.qvalues[0], network_value=tree.raw_values[0, root],
                         proven_wins=(tree.children_visits[0, root] > 0)
                         & (tree.children_rewards[0, root] == 1)
                         & (tree.children_discounts[0, root] == 0)
                         & legal_actions(state),
                         rewards=tree.children_rewards[0, root],
                         discounts=tree.children_discounts[0, root]))


class AlphaZeroAgent:
    def __init__(self, model, simulations=128, c_puct=1.5, search_policy="puct", *,
                 proven_win_priority=False, prior_temperature=1.0, gumbel_q_scale=0.1):
        self.device = jax.devices("cpu")[0]
        self.model = jax.device_put(model, self.device)
        self.settings = SearchConfig(simulations=simulations, c_puct=c_puct, policy=search_policy,
                                     proven_win_priority=proven_win_priority,
                                     prior_temperature=prior_temperature, gumbel_q_scale=gumbel_q_scale)
        self.last_result = None
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path=DEFAULT_MODEL_PATH, simulations=128, c_puct=1.5, search_policy=None, *,
                        proven_win_priority=None, prior_temperature=None, gumbel_q_scale=None):
        model, metadata = load_checkpoint(path, jax.devices("cpu")[0])
        settings = SearchConfig.from_metadata(metadata, simulations=simulations, c_puct=c_puct,
                                              policy=search_policy, proven_win_priority=proven_win_priority,
                                              prior_temperature=prior_temperature, gumbel_q_scale=gumbel_q_scale)
        return cls(model, simulations, c_puct, settings.policy,
                   proven_win_priority=settings.proven_win_priority,
                   prior_temperature=settings.prior_temperature, gumbel_q_scale=settings.gumbel_q_scale)

    def predict(self, board, to_play):
        with jax.default_device(self.device):
            state = from_board(board, to_play, self.model.config)
            logits, value = _predict(self.model, state)
            return np.asarray(logits), float(value)

    def warmup(self):
        board = np.zeros((self.model.config.rows, self.model.config.cols), np.int8)
        self.get_move(board, raw=True)
        self.get_move(board)

    def get_move(self, board, my_piece=1, opp_piece=2, raw=False, *, diagnostics=False):
        if my_piece not in (1, 2) or opp_piece != 3 - my_piece:
            raise ValueError("AlphaZero uses pieces 1 and 2")
        with self._lock, jax.default_device(self.device):
            state = from_board(board, my_piece, self.model.config)
            if bool(state.done):
                self.last_result = None
                return None
            legal = np.asarray(legal_actions(state))
            if raw:
                logits, value = _predict(self.model, state)
                action = int(np.argmax(np.where(legal, np.asarray(logits), -np.inf)))
                self.last_result = (dict(action=action, legal=legal, network_logits=np.asarray(logits),
                                         network_value=float(value), mode="raw") if diagnostics else None)
                return action
            output = (_search(self.model, state, self.settings, True) if diagnostics else
                      _search(self.model, state, self.settings))
            policy, visits, value, action = output[:4]
            action = int(action)
            if not legal[action]:
                raise RuntimeError("Search returned an illegal action")
            self.last_result = dict(policy=np.asarray(policy), visits=np.asarray(visits), value=float(value),
                                    action=action, search_policy=self.settings.policy,
                                    proven_win_priority=self.settings.proven_win_priority,
                                    prior_temperature=self.settings.prior_temperature,
                                    gumbel_q_scale=self.settings.gumbel_q_scale)
            if diagnostics:
                self.last_result.update({name: np.asarray(item) for name, item in output[4].items()})
                self.last_result.update(legal=legal, mode="search")
            return action
