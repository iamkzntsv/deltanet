import chex
import jax
import jax.numpy as jnp
import pytest
from conftest import D_KEY, D_VALUE, T

from deltanet.chunkwise import compute_chunkwise_state, compute_wy
from deltanet.recurrent import compute_recurrent_state


def test_wy_shapes(make_inputs_sequence):
    K, V, _, beta = make_inputs_sequence()

    W, U = compute_wy(K, V, beta)

    chex.assert_shape(W, (T, D_KEY))
    chex.assert_shape(U, (T, D_VALUE))


def test_wy_computation(make_inputs_sequence):
    def _compute_Aj(j):
        k = K[j, :]
        b = beta[j, :]
        return jnp.eye(D_KEY) - b * jnp.outer(k, k)

    K, V, _, beta = make_inputs_sequence()

    # Compute A
    A = jnp.eye(D_KEY)
    for i in range(T - 1, -1, -1):
        A = A @ _compute_Aj(i)

    # Compute B
    B = jnp.zeros((D_KEY, D_VALUE))
    for i in range(T):
        B_i = beta[i, :] * jnp.outer(K[i, :], V[i, :])

        A_B = jnp.eye(D_KEY)
        for j in range(T - 1, i, -1):
            A_B = A_B @ _compute_Aj(j)

        B += A_B @ B_i

    W, U = compute_wy(K, V, beta)

    A_wy = jnp.eye(D_KEY) - K.T @ W
    B_wy = K.T @ U

    chex.assert_trees_all_close(A, A_wy, atol=1e-3)
    chex.assert_trees_all_close(B, B_wy, atol=1e-3)


@pytest.mark.parametrize("C", [1, 2, 4, 8, 16])
def test_chunk_sizes(make_inputs_sequence, C):
    K, V, Q, beta = make_inputs_sequence()
    S_rec_chunk, O_rec_chunk = compute_recurrent_state(K[:C], V[:C], Q[:C], beta[:C])
    S_cw_chunk, O_cw_chunk = compute_chunkwise_state(
        K[:C, :], V[:C, :], Q[:C, :], beta[:C], C
    )

    chex.assert_trees_all_close(S_rec_chunk, S_cw_chunk, atol=1e-5)
    chex.assert_trees_all_close(O_rec_chunk, O_cw_chunk, atol=1e-5)


@pytest.mark.parametrize("C", [4, 8])
def test_chunkwise_outputs(make_inputs_sequence, C):
    K, V, Q, beta = make_inputs_sequence()

    _, O_rec = compute_recurrent_state(K, V, Q, beta)
    _, O_cw = compute_chunkwise_state(K, V, Q, beta, C)

    chex.assert_trees_all_close(O_rec, O_cw, atol=1e-5)


@pytest.mark.parametrize("C", [4, 8])
def test_chunkwise_backward(make_inputs_sequence, C):
    def _dummy_loss(forward_fn, K, V, Q, beta, *args):
        _, O = forward_fn(K, V, Q, beta, *args)

        y = jax.random.normal(jax.random.PRNGKey(42), O.shape)

        return jnp.sum(jnp.square(O - y)) * 0.5

    K, V, Q, beta = make_inputs_sequence()

    grad_fn = jax.grad(_dummy_loss, argnums=(1, 2, 3, 4))

    grads_rec = grad_fn(compute_recurrent_state, K, V, Q, beta)
    grads_cw = grad_fn(compute_chunkwise_state, K, V, Q, beta, C)

    chex.assert_trees_all_close(grads_rec, grads_cw, atol=1e-5)
