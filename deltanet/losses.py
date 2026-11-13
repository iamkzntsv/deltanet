import jax.numpy as jnp
import optax


def language_model_loss(model, x, y):
    """Simple cross-entropy loss"""
    logits = model(x)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits


def associative_recall_loss(model, x, y, ignore_index=-100):
    """Cross-entropy loss with position masking"""
    logits = model(x)

    mask = y != ignore_index
    targets = jnp.where(mask, y, 0)

    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, targets)

    return (loss * mask).sum() / mask.sum(), logits
