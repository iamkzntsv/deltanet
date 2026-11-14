import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    from deltanet.recurrent import compute_recurrent_state
    from deltanet.utils import new_key, normalize

    steps = [
        1,
        50,
        100,
        1000,
        10_000,
        100_000,
    ]

    T = max(steps)
    d_key = 8
    d_query = 8
    d_value = 8

    key_k, key_v, key_q, _ = new_key(split=4)

    keys = jax.jit(normalize)(jax.random.normal(key_k, (T, d_key)))
    queries = jax.jit(normalize)(jax.random.normal(key_q, (T, d_query)))
    values = jax.jit(normalize)(jax.random.normal(key_v, (T, d_value)))
    betas = jax.jit(normalize)(jnp.ones((T, 1)))

    k0, v0 = normalize(keys[0, :]), normalize(values[0, :])

    for s in steps:
        updated_state, _ = compute_recurrent_state(
            keys[:s, :], values[:s, :], queries[:s, :], betas[:s]
        )
        o = k0 @ updated_state
        err = jnp.sum(jnp.square(o - v0))
        print(f"---steps={s}, retrieval error: {err}---")


if __name__ == "__main__":
    app.run()
