import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from deltanet.model import DeltaNet
from deltanet.tokenizer import get_tokenizer
from deltanet.utils import sample_batch

logger = logging.getLogger(__name__)


def loss_fn(model: DeltaNet, x: jax.Array, y: jax.Array):
    return optax.losses.softmax_cross_entropy_with_integer_labels(model(x), y).mean()


@nnx.jit
def train_step(model: DeltaNet, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)

    return loss


def train(model, optimizer, x, y, cfg, ckpt_mngr):
    for step in range(cfg.num_steps + 1):
        loss = train_step(model, optimizer, x, y)

        if step % 100 == 0 or step == cfg.num_steps:
            logger.info(f"Step: {step}, Loss: {loss.item()}")

            _graphdef, state = nnx.split(model)

            ckpt_mngr.save(step, args=ocp.args.StandardSave(state))
            ckpt_mngr.wait_until_finished()

    ckpt_mngr.close()


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model
    train_cfg = cfg.training

    tokenizer = get_tokenizer(model_cfg.tokenizer)(train_cfg.data_path)

    with open(train_cfg.data_path, encoding="utf-8") as f:
        tokens = jnp.asarray(tokenizer.encode(f.read()))

    rngs = nnx.Rngs(cfg.seed)
    model = DeltaNet(**model_cfg.model, vocab_size=tokenizer.vocab_size, rngs=rngs)
    optimizer = nnx.Optimizer(
        model, tx=optax.adamw(**train_cfg.optimizer), wrt=nnx.Param
    )

    x, y = sample_batch(
        tokens, train_cfg.batch_size, train_cfg.seq_len, jax.random.PRNGKey(0)
    )

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    train(model, optimizer, x, y, train_cfg, ckpt_mngr)


if __name__ == "__main__":
    main()
