import math

import jax
import jax.numpy as jnp
from flax import nnx


def normalize(x: jax.Array, eps: float = 1e-6):
    return x / jnp.maximum(jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True), eps)


def new_key(split: int = 1):
    key = jax.random.PRNGKey(0)
    return jax.random.split(key, split)


def sample_batch(
    tokens: jax.Array, batch_size: int, seq_len: int, key: jax.random.PRNGKey
):
    starts = jax.random.randint(
        key=key, shape=(batch_size,), minval=0, maxval=len(tokens) - seq_len
    )

    offsets = jnp.arange(seq_len)
    positions = starts[:, None] + offsets[None, :]

    x = tokens[positions]
    y = tokens[positions + 1]

    return x, y


class SwiGLU(nnx.Module):
    def __init__(self, d_in: int, d_hidden: int | None = None, *, rngs: nnx.Rngs):
        d_hidden = d_hidden or self._compute_hidden_dim(d_in)
        self.gate_proj = nnx.Linear(d_in, d_hidden, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(d_in, d_hidden, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(d_hidden, d_in, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nnx.silu(gate) * up)

    def _compute_hidden_dim(self, d_in: int):
        return math.ceil(8 / 3 * d_in)
