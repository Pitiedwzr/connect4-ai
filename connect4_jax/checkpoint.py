"""Atomic, portable bundles: JSON metadata, Equinox leaves, and NumPy replay.

Inference loading never reads optimizer/replay data and requires no PyTorch.
"""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .config import Config
from .network import PolicyValueNet

DEFAULT_MODEL_PATH = "model/connect4_alphazero.eqx"


def _leaves(tree):
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, tree)
    return stream.getvalue()


def _validate_weights(model):
    if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(model)
           if eqx.is_inexact_array(x)):
        raise ValueError("Checkpoint model weights must be finite")


def save_checkpoint(path, model, *, metadata=None, optimizer_state=None, replay=None):
    _validate_weights(model)
    payload = dict(metadata or {})
    payload.update(format_version=1, algorithm="alphazero_eqx", config=asdict(model.config))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("metadata.json", json.dumps(payload))
                bundle.writestr("model.eqx", _leaves(model))
                if optimizer_state is not None:
                    bundle.writestr("optimizer.eqx", _leaves(optimizer_state))
                if replay is not None:
                    data = io.BytesIO()
                    np.savez_compressed(data, **replay.arrays())
                    bundle.writestr("replay.npz", data.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device=None):
    try:
        with zipfile.ZipFile(path) as bundle:
            payload = json.loads(bundle.read("metadata.json"))
            if payload.get("format_version") != 1 or payload.get("algorithm") != "alphazero_eqx":
                raise ValueError("Expected a version-1 Equinox AlphaZero checkpoint")
            config = Config(**payload["config"])
            skeleton = eqx.filter_eval_shape(PolicyValueNet, config, jax.random.PRNGKey(0))
            model = eqx.tree_deserialise_leaves(io.BytesIO(bundle.read("model.eqx")), skeleton)
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, EOFError) as exc:
        raise ValueError(f"Invalid Equinox AlphaZero checkpoint: {path}") from exc
    model = jax.tree.map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, model)
    if device is not None:
        model = jax.device_put(model, device)
    _validate_weights(model)
    return model, payload


def restore_training(path, optimizer_like, replay):
    with zipfile.ZipFile(path) as bundle:
        if "optimizer.eqx" not in bundle.namelist() or "replay.npz" not in bundle.namelist():
            raise ValueError("Inference-only checkpoint cannot resume training")
        optimizer_state = eqx.tree_deserialise_leaves(
            io.BytesIO(bundle.read("optimizer.eqx")), optimizer_like)
        with np.load(io.BytesIO(bundle.read("replay.npz")), allow_pickle=False) as data:
            replay.restore(data)
    return optimizer_state
