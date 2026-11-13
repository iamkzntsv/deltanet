"""Multi-query associative recall (MQAR) data.

Adapted from HazyResearch/zoology (zoology/data/associative_recall.py).
"""

import numpy as np


def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (inputs, labels), each of shape (num_examples, input_seq_len).

    Labels are -100 everywhere except at query positions, where they hold the value
    bound to the queried key.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    rng = np.random.default_rng(seed)
    context_size = num_kv_pairs * 2 * num_passes

    # keys from the lower half of the vocab, values from the upper half; unique per example
    key_vocab_size = vocab_size // 2
    keys = np.stack(
        [
            rng.choice(np.arange(1, key_vocab_size), size=num_kv_pairs, replace=False)
            for _ in range(num_examples)
        ]
    )
    values = np.stack(
        [
            rng.choice(
                np.arange(key_vocab_size, vocab_size), size=num_kv_pairs, replace=False
            )
            for _ in range(num_examples)
        ]
    )

    kvs = np.zeros((num_examples, num_kv_pairs * 2), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    # power-law distributed gaps between the context and each query
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack(
        [
            rng.choice(space, size=num_kv_pairs, replace=False, p=p)
            for _ in range(num_examples)
        ]
    )

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values, axis=1)

    inputs, labels = examples[:, :-1], labels[:, 1:]

    if random_non_queries:
        mask = inputs == 0
        inputs[mask] = rng.integers(vocab_size, size=mask.sum())

    return inputs, labels
