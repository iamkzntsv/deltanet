from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from deltanet.model import DeltaNet
from deltanet.tokenizer import CharTokenizer
from deltanet.utils import new_key, normalize

DATA_PATH = Path(__file__).parent.parent / "data" / "shakespeare.txt"
D_KEY = D_VALUE = 8
D_QUERY = 8
T = 128
B = 16


@pytest.fixture
def make_inputs_single_token():
    def _make():
        key_S, key_k, key_v, key_query = new_key(4)

        S = normalize(jax.random.normal(key_S, (D_KEY, D_VALUE)))
        k = normalize(jax.random.normal(key_k, (D_KEY,)))
        v = normalize(jax.random.normal(key_v, (D_VALUE,)))
        q = normalize(jax.random.normal(key_query, (D_QUERY,)))

        return S, k, v, q

    return _make


@pytest.fixture
def make_inputs_sequence():
    def _make(T=T):
        key_k, key_v, key_q, key_betas = new_key(4)

        K = jax.random.normal(key_k, (T, D_KEY))
        V = jax.random.normal(key_v, (T, D_VALUE))
        Q = jax.random.normal(key_q, (T, D_QUERY))
        betas = jax.random.uniform(key_betas, (T, 1))

        K = jax.vmap(normalize)(K)
        Q = jax.vmap(normalize)(Q)
        V = jax.vmap(normalize)(V)

        return K, V, Q, betas

    return _make


@pytest.fixture(scope="session")
def tokens(make_deltanet):
    tkn, _ = make_deltanet()
    text = DATA_PATH.read_text(encoding="utf-8")[:10_000]
    return jnp.array(tkn.encode(text))


@pytest.fixture
def make_batch(tokens):
    def _make(B=B, T=T, offset=0):
        n = B * T
        return tokens[offset : offset + n].reshape(B, T)

    return _make


@pytest.fixture(scope="session")
def make_deltanet(**kwargs):
    def _make(**kwargs):
        tkn = CharTokenizer("data/shakespeare.txt")
        vocab_size = tkn.vocab_size

        mdl = DeltaNet(
            vocab_size=vocab_size,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            conv_kernel_size=4,
            chunk_size=4,
            **kwargs,
            rngs=nnx.Rngs(0),
        )

        return tkn, mdl

    return _make
