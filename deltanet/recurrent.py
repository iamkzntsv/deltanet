from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def delta_rule(S, k, v):
    return jnp.sum((k @ S - v) ** 2) * 0.5


@jax.jit
def delta_step(S_in, k, v, q, beta=1):
    """
    Single token update
    k:        (d_k,)
    S_in:   (d_k, d_v)
    v:        (d_v,)
    q:        (d_k,)
    beta:     scalar
    """
    v_old = k @ S_in
    v_new = beta * v + (1 - beta) * v_old

    S = S_in - jnp.outer(k, v_old) + jnp.outer(k, v_new)
    o = q @ S

    return S, o


@partial(jax.jit, static_argnums=4)
def compute_recurrent_state(K, V, Q, betas, return_state=True):
    """
    keys: (T, d_k)
    values: (T, d_v)
    """

    def _scan_fn(state: jax.Array, inputs: tuple):
        k, v, q, b = inputs
        updated_state, o = delta_step(state, k, v, q, b)
        return updated_state, o

    d_k, d_v = K.shape[-1], V.shape[-1]

    S_in = jnp.zeros((d_k, d_v), dtype=V.dtype)

    S_final, outputs = jax.lax.scan(
        _scan_fn,
        S_in,
        (K, V, Q, betas),
    )

    return (S_final, outputs) if return_state else outputs
