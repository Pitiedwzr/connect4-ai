"""Bounded host replay with reflection augmentation at sampling time."""
import numpy as np


class Replay:
    def __init__(self, capacity, config):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.states = np.empty((capacity, 2, config.rows, config.cols), np.uint8)
        self.policies = np.empty((capacity, config.cols), np.float32)
        self.outcomes = np.empty(capacity, np.float32)
        self.size = self.cursor = 0

    def __len__(self):
        return self.size

    def add(self, trajectory):
        valid = np.asarray(trajectory.valid, dtype=bool)
        states = np.asarray(trajectory.states)[valid]
        policies = np.asarray(trajectory.policies)[valid]
        outcomes = np.asarray(trajectory.outcomes)[valid]
        legal = ~states[:, :, -1, :].any(axis=1)
        if (not np.isfinite(policies).all() or not np.isfinite(outcomes).all()
                or (policies < 0).any() or (policies[~legal] != 0).any()
                or not np.allclose(policies.sum(axis=-1), 1.0, atol=1e-5)
                or not np.isin(outcomes, (-1, 0, 1)).all()):
            raise ValueError("Replay requires finite legal policy distributions and final outcomes")
        # Keep the most recent positions if a collection exceeds capacity.
        states, policies, outcomes = (x[-self.capacity:] for x in (states, policies, outcomes))
        count = len(outcomes)
        indices = (self.cursor + np.arange(count)) % self.capacity
        self.states[indices], self.policies[indices], self.outcomes[indices] = states, policies, outcomes
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size, rng):
        if not self.size:
            raise ValueError("Cannot sample empty replay")
        indices = rng.integers(self.size, size=batch_size)
        states = self.states[indices].astype(np.float32)
        policies = self.policies[indices].copy()
        reflect = rng.random(batch_size) < 0.5
        states[reflect] = states[reflect, :, :, ::-1]
        policies[reflect] = policies[reflect, ::-1]
        return states, policies, self.outcomes[indices].copy()

    def arrays(self):
        return dict(states=self.states[:self.size], policies=self.policies[:self.size],
                    outcomes=self.outcomes[:self.size], cursor=np.asarray(self.cursor))

    def restore(self, arrays):
        count = len(arrays["outcomes"])
        if count > self.capacity:
            raise ValueError("Saved replay exceeds configured capacity")
        for name in ("states", "policies", "outcomes"):
            target, saved = getattr(self, name), arrays[name]
            if saved.shape != target[:count].shape:
                raise ValueError("Saved replay dimensions do not match configuration")
            target[:count] = saved
        self.size = count
        self.cursor = int(arrays["cursor"])
        if not 0 <= self.cursor < self.capacity or (count < self.capacity and self.cursor != count):
            raise ValueError("Invalid saved replay cursor")
