import chex
import jax
import jax.numpy as jnp
from conftest import D_KEY, D_VALUE

from deltanet.recurrent import compute_recurrent_state, delta_rule, delta_step
from deltanet.utils import normalize


def test_grads_wrt_state_matches_analytic_form(make_inputs_single_token):
    S, k, v, _ = make_inputs_single_token()
    grad_auto = jax.grad(delta_rule, argnums=0)(S, k, v)
    grad_analytic = jnp.outer(k, (k @ S - v))

    chex.assert_trees_all_close(grad_auto, grad_analytic)


def test_delta_step_update_shape(make_inputs_single_token):
    S_in, k, v, q = make_inputs_single_token()
    S, o = delta_step(S_in, k, v, q)
    chex.assert_shape(S, S_in.shape)
    chex.assert_shape(o, v.shape)


def test_state_consistency_with_zero_beta(make_inputs_sequence):
    keys, values, queries, beta = make_inputs_sequence()
    beta *= 0

    final_state, _ = compute_recurrent_state(keys, values, queries, beta)

    chex.assert_trees_all_close(final_state, jnp.zeros_like(final_state))


def test_reccurent_loop_scan_verify(make_inputs_sequence):
    K, V, Q, beta = make_inputs_sequence()

    T = K.shape[0]
    d_k, d_v = K.shape[-1], V.shape[-1]

    loop_state = jnp.zeros((d_k, d_v))
    for i in range(T):
        loop_state, _ = delta_step(loop_state, K[i, :], V[i, :], Q[i, :], beta[i, :])

    scan_final_state, _ = compute_recurrent_state(K, V, Q, beta)

    chex.assert_trees_all_close(scan_final_state, loop_state, atol=1e-5)


def test_normalized_key_value_retrieval(make_inputs_sequence):
    """Check if for beta=1 and norm(k)=1 querying with the same k retrieves v"""
    K, V, Q, _ = make_inputs_sequence()
    k1, v1, q1 = (
        normalize(K[0, :]),
        normalize(V[0, :]),
        normalize(Q[0, :]),
    )

    S_in = jnp.zeros((D_KEY, D_VALUE))
    S_final, _ = delta_step(S_in, k1, v1, q1)

    chex.assert_trees_all_close(v1, k1 @ S_final)
