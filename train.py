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

from deltanet.losses import associative_recall_loss, language_model_loss
from deltanet.model import DeltaNet
from deltanet.tokenizer import CharTokenizer
from deltanet.trainer import train

logger = logging.getLogger(__name__)


TOKENIZERS = {"character": CharTokenizer}

LOSSES = {
    "language_modeling": language_model_loss,
    "associative_recall": associative_recall_loss,
}


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model
    train_cfg = cfg.training

    tokenizer = TOKENIZERS[model_cfg.tokenizer](train_cfg.data_path)
    loss_fn = LOSSES[cfg.objective]

    with open(train_cfg.data_path, encoding="utf-8") as f:
        tokens = jnp.asarray(tokenizer.encode(f.read()))

    rngs = nnx.Rngs(cfg.seed)
    model = DeltaNet(**model_cfg.model, vocab_size=tokenizer.vocab_size, rngs=rngs)
    optimizer = nnx.Optimizer(
        model, tx=optax.adamw(**train_cfg.optimizer), wrt=nnx.Param
    )

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    # Generate Shakespeare data
    starts = jax.random.randint(
        key=jax.random.PRNGKey(0),
        shape=(train_cfg.batch_size,),
        minval=0,
        maxval=len(tokens) - train_cfg.seq_len,
    )

    offsets = jnp.arange(train_cfg.seq_len)
    positions = starts[:, None] + offsets[None, :]

    x = tokens[positions]
    y = tokens[positions + 1]

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        x=x,
        y=y,
        cfg=train_cfg,
        ckpt_mngr=ckpt_mngr,
    )


if __name__ == "__main__":
    main()
