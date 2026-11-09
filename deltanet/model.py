from functools import partial

import einops
import jax
import jax.numpy as jnp
from flax import nnx

from deltanet.chunkwise import compute_chunkwise_state
from deltanet.recurrent import compute_recurrent_state
from deltanet.utils import SwiGLU as swiglu
from deltanet.utils import normalize


class DeltaNetBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        d_key: int,
        d_value: int,
        d_query: int,
        conv_kernel_size: int,
        chunk_size: int,
        *,
        use_cw: bool = True,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        d_key_mh = d_key * num_heads
        d_value_mh = d_value * num_heads
        d_query_mh = d_query * num_heads

        self.k_proj = nnx.Linear(
            in_features=embed_dim, out_features=d_key_mh, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            in_features=embed_dim, out_features=d_value_mh, rngs=rngs
        )
        self.q_proj = nnx.Linear(
            in_features=embed_dim, out_features=d_query_mh, rngs=rngs
        )
        self.W_beta = nnx.Linear(
            in_features=embed_dim, out_features=num_heads, rngs=rngs
        )

        self.k_conv = nnx.Conv(
            in_features=d_key_mh,
            out_features=d_key_mh,
            kernel_size=(conv_kernel_size,),
            feature_group_count=d_key_mh,
            padding="CAUSAL",
            use_bias=False,
            rngs=rngs,
        )

        self.q_conv = nnx.Conv(
            in_features=d_query_mh,
            out_features=d_query_mh,
            kernel_size=(conv_kernel_size,),
            feature_group_count=d_query_mh,
            padding="CAUSAL",
            use_bias=False,
            rngs=rngs,
        )

        self.v_conv = nnx.Conv(
            in_features=d_value_mh,
            out_features=d_value_mh,
            kernel_size=(conv_kernel_size,),
            feature_group_count=d_value_mh,
            padding="CAUSAL",
            use_bias=False,
            rngs=rngs,
        )

        self.forward_fn = (
            partial(compute_chunkwise_state, C=self.chunk_size, return_state=False)
            if use_cw
            else partial(compute_recurrent_state, return_state=False)
        )

        self.o_norm = nnx.RMSNorm(d_value, rngs=rngs)
        self.o_proj = nnx.Linear(
            in_features=self.num_heads * d_value, out_features=embed_dim, rngs=rngs
        )

    def __call__(self, x: jax.Array):
        K = nnx.silu(self.k_conv(self.k_proj(x)))
        Q = nnx.silu(self.q_conv(self.q_proj(x)))
        V = nnx.silu(self.v_conv(self.v_proj(x)))
        beta = nnx.sigmoid(self.W_beta(x))

        K = einops.rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        Q = einops.rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        V = einops.rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)
        beta = einops.rearrange(beta, "b t h -> b h t 1")

        K, Q = normalize(K), normalize(Q)

        # vmap over heads then batch
        O = jax.vmap(jax.vmap(self.forward_fn))(K, V, Q, beta)

        O = einops.rearrange(
            self.o_norm(O),  # normalize over v
            "b h t v -> b t (h v)",
        )

        return self.o_proj(O)


class DeltaNetLayer(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_head: int,
        conv_kernel_size: int,
        chunk_size: int,
        *,
        use_cw: bool = True,
        rngs: nnx.Rngs,
    ):
        self.rms_norm1 = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.mixer = DeltaNetBlock(
            num_heads=num_heads,
            embed_dim=embed_dim,
            d_key=d_head,
            d_value=d_head,
            d_query=d_head,
            conv_kernel_size=conv_kernel_size,
            chunk_size=chunk_size,
            use_cw=use_cw,
            rngs=rngs,
        )
        self.rms_norm2 = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.swiglu = swiglu(d_in=embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = x + self.mixer(self.rms_norm1(x))
        x = x + self.swiglu(self.rms_norm2(x))
        return x


class DeltaNet(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        conv_kernel_size: int,
        chunk_size: int,
        use_cw: bool = True,
        *,
        rngs: nnx.Rngs,
    ):

        assert embed_dim % num_heads == 0

        self.vocab_size = vocab_size
        self.embed = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.layers = nnx.List(
            [
                DeltaNetLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    d_head=embed_dim // num_heads,
                    conv_kernel_size=conv_kernel_size,
                    chunk_size=chunk_size,
                    use_cw=use_cw,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm_final = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(
            in_features=embed_dim, out_features=vocab_size, rngs=rngs
        )

    def __call__(self, x: jax.Array):

        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)

        return self.out_proj(self.rms_norm_final(x))


if __name__ == "__main__":
    from deltanet.tokenizer import CharTokenizer

    tokenizer = CharTokenizer("data/shakespeare.txt")
    inputs = jnp.array(tokenizer.encode("The cat sat on a mat"))

    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    conv_kernel_size = 4
    chunk_size = 4

    rngs = nnx.Rngs(0)

    inputs_batch = jnp.array([inputs] * 10)

    mdl = DeltaNet(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    mdl(inputs_batch)
