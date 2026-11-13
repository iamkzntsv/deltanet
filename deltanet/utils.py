import math

import jax
import jax.numpy as jnp
from flax import nnx


def normalize(x: jax.Array, eps: float = 1e-6):
    return x / jnp.maximum(jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True), eps)


def new_key(split: int = 1):
    key = jax.random.PRNGKey(0)
    return jax.random.split(key, split)


def get_batch(key: jax.random.PRNGKey, batch_size, x, y):
    idx = jax.random.randint(key, (batch_size,), 0, len(x))
    return x[idx], y[idx]


class MaskedAccuracy(nnx.metrics.Average):
    """Used for evaluating in associative recall tasks"""

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def update(self, logits, labels, **_):
        mask = labels != self.ignore_index
        super().update(values=(logits.argmax(-1) == labels), mask=mask)


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
