"""Import legacy AlphaZero weights; optimizer and replay start afresh."""
import argparse
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from connect4_jax.checkpoint import save_checkpoint
from connect4_jax.config import Config
from connect4_jax.network import PolicyValueNet


def convert_model(legacy):
    config = Config(**asdict(legacy.config))
    model = PolicyValueNet(config, jax.random.PRNGKey(0))

    def copy_layer(new, old):
        if tuple(old.weight.shape) != tuple(new.weight.shape):
            raise ValueError("Layer shapes do not match")
        new = eqx.tree_at(lambda layer: layer.weight, new,
                          jnp.asarray(old.weight.detach().cpu().numpy()))
        if old.bias is not None:
            # Equinox convolution biases broadcast over spatial dimensions.
            bias = jnp.asarray(old.bias.detach().cpu().numpy()).reshape(new.bias.shape)
            new = eqx.tree_at(lambda layer: layer.bias, new, bias)
        return new

    model = eqx.tree_at(lambda m: (m.stem, m.norm), model,
                        (copy_layer(model.stem, legacy.trunk[0]), copy_layer(model.norm, legacy.trunk[1])))
    for i, block in enumerate(model.blocks):
        old = legacy.trunk[i + 3]
        block = eqx.tree_at(lambda b: (b.conv1, b.norm1, b.conv2, b.norm2), block,
                            tuple(copy_layer(getattr(block, name), getattr(old, name))
                                  for name in ("conv1", "norm1", "conv2", "norm2")))
        model = eqx.tree_at(lambda m: m.blocks[i], model, block)
    for name in ("policy_conv", "policy_fc", "value_conv", "value_fc", "value_out"):
        model = eqx.tree_at(lambda m: getattr(m, name), model,
                            copy_layer(getattr(model, name), getattr(legacy, name)))
    return model


def main():
    import torch
    from alphazero import load_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    legacy, _ = load_checkpoint(args.input, "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        model = convert_model(legacy)
        inputs = np.random.default_rng(42).integers(0, 2, (8, 2, model.config.rows, model.config.cols)).astype(np.float32)
        with torch.no_grad():
            expected_logits, expected_value = legacy(torch.from_numpy(inputs))
        logits, values = eqx.filter_jit(jax.vmap(model))(jnp.asarray(inputs))
        np.testing.assert_allclose(np.asarray(logits), expected_logits.numpy(), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(np.asarray(values), expected_value.numpy().reshape(-1), atol=2e-5, rtol=2e-5)
        save_checkpoint(args.output, model, metadata=dict(converted_from=args.input))
    print(f"Verified forward parity and saved {args.output}; use train_jax.py --init-from to train")


if __name__ == "__main__":
    main()
