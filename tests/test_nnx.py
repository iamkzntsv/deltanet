import chex
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


def test_chunkwise_vs_recurrent(make_deltanet, make_batch):
    batch = make_batch()

    _, model_cw = make_deltanet()
    _, model_rec = make_deltanet(use_cw=False)

    O_cw = model_cw(batch)
    O_rec = model_rec(batch)

    chex.assert_trees_all_close(O_cw, O_rec, atol=1e-4)


def test_token_causality(make_deltanet, tokens):
    _, model = make_deltanet()
    toks = tokens[jnp.newaxis, :]
    last_tok_idx = toks.shape[1] - 1
    toks_modified = toks.at[0, last_tok_idx].set(
        (toks[0, last_tok_idx] + 1) % model.vocab_size
    )

    logits_original = model(toks)
    logits_modified = model(toks_modified)

    chex.assert_trees_all_close(
        logits_original[:, :last_tok_idx], logits_modified[:, :last_tok_idx]
    )


@pytest.mark.parametrize("T", [64])
def test_gradient_flow(make_deltanet, make_batch, T):
    def loss_fn(mdl, x, y):
        logits = mdl(x)
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        return loss, logits

    _, model = make_deltanet()

    full_batch = make_batch()
    x = full_batch[:, :T]
    y = full_batch[:, 1 : T + 1]

    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x, y)

    chex.assert_shape(y, logits.shape[:-1])

    chex.assert_type(y, jnp.int32)
    chex.assert_type(logits, jnp.float32)

    chex.assert_tree_no_nones(loss)
    chex.assert_tree_no_nones(grads)
