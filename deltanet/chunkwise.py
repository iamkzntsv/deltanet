"""
Implements a WU parallel computation for a single chunk
"""

from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def compute_wy(K, V, beta):
    """
    K: (T, d_k)
    V: (T, d_v)
    beta: (T, 1)
    """
    G = K @ K.T
    L = jnp.tril(G, k=-1)

    C = K.shape[0]
    A = jnp.eye(C) + beta * L
    W = jax.scipy.linalg.solve_triangular(A, beta * K, lower=True)

    U = jax.scipy.linalg.solve_triangular(A, beta * V, lower=True)

    return W, U


@jax.jit
def chunkwise_forward(S_in, K, V, Q, beta):
    W, U = compute_wy(K, V, beta)
    V_tilde = U - W @ S_in

    S = K.T @ V_tilde + S_in

    C = K.shape[0]
    M = jnp.tril(jnp.ones((C, C), dtype=K.dtype))
    O = Q @ S_in + jnp.multiply(Q @ K.T, M) @ V_tilde

    return S, O


@partial(jax.jit, static_argnums=(4, 5))
def compute_chunkwise_state(K, V, Q, beta, C, return_state=True):
    T = K.shape[0]
    pad = (-T) % C

    def _prepare(A):
        """Reshape and pad"""
        A = jnp.pad(A, ((0, pad), (0, 0)))
        return jnp.reshape(A, (-1, C, A.shape[-1]))

    def _scan_fn(S_in, inputs):
        S_upd, O = chunkwise_forward(S_in, *inputs)
        return S_upd, O

    K = _prepare(K)
    V = _prepare(V)
    Q = _prepare(Q)
    beta = _prepare(beta)

    S_in = jnp.zeros((K.shape[-1], V.shape[-1]), dtype=K.dtype)
    S_T, O = jax.lax.scan(_scan_fn, S_in, (K, V, Q, beta))

    # reshape and remove padded positions
    O = jnp.reshape(O, (-1, V.shape[-1]))[:T]

    return (S_T, O) if return_state else O
