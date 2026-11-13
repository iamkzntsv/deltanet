import logging

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from deltanet.utils import MaskedAccuracy, get_batch

logger = logging.getLogger(__name__)


@nnx.jit(static_argnames="loss_fn")
def train_step(model, optimizer, x, y, loss_fn):
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    optimizer.update(model, grads)

    return loss, logits


@nnx.jit(static_argnames="loss_fn")
def eval_step(model, metrics, loss_fn, x, y):
    loss, logits = loss_fn(model, x, y)

    metrics.update(loss=loss, logits=logits, labels=y)


def evaluate(model, metrics, loss_fn, x, y, batch_size=16):
    metrics.reset()
    for i in range(0, len(x), batch_size):
        eval_step(model, metrics, loss_fn, x[i : i + batch_size], y[i : i + batch_size])

    return metrics.compute()


def train(model, optimizer, loss_fn, train_dataset, eval_dataset, cfg, ckpt_mngr, seed):

    x_train, y_train = (jnp.asarray(a, dtype=jnp.int32) for a in train_dataset)
    x_test, y_test = (jnp.asarray(a, dtype=jnp.int32) for a in eval_dataset)

    train_metrics = nnx.MultiMetric(
        accuracy=MaskedAccuracy(), loss=nnx.metrics.Average("loss")
    )

    test_metrics = nnx.MultiMetric(
        accuracy=MaskedAccuracy(), loss=nnx.metrics.Average("loss")
    )

    key = jax.random.PRNGKey(seed)
    for step in range(cfg.num_steps + 1):
        key = jax.random.fold_in(key, step)
        (
            x_batch,
            y_batch,
        ) = get_batch(key, cfg.batch_size, x_train, y_train)

        loss, logits = train_step(model, optimizer, x_batch, y_batch, loss_fn)
        train_metrics.update(loss=loss, logits=logits, labels=y_batch)

        if step % cfg.eval_every == 0 or step == cfg.num_steps:
            train_res = train_metrics.compute()
            train_metrics.reset()
            test_res = evaluate(
                model, test_metrics, loss_fn, x_test, y_test, cfg.batch_size
            )

            logger.info(
                f"Step {step} | "
                f"train loss {train_res['loss']:.4f} acc: {train_res['accuracy']:.4f} | "
                f"test loss {test_res['loss']:.4f} acc: {test_res['accuracy']:.4f}"
            )

            _graphdef, state = nnx.split(model)

            ckpt_mngr.save(step, args=ocp.args.StandardSave(state))
            ckpt_mngr.wait_until_finished()

    ckpt_mngr.close()
