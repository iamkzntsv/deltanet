import logging
from pathlib import Path

import hydra
import optax
import orbax.checkpoint as ocp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import deltanet.config  # noqa: F401
from deltanet.losses import associative_recall_loss, language_model_loss
from deltanet.model import DeltaNet
from deltanet.tokenizer import CharTokenizer
from deltanet.trainer import train
from deltanet.utils import create_shakespeare

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

    rngs = nnx.Rngs(cfg.seed)
    model = DeltaNet(**model_cfg.model, vocab_size=tokenizer.vocab_size, rngs=rngs)

    lr_schedule = optax.warmup_cosine_decay_schedule(**train_cfg.optimizer.lr)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, **train_cfg.optimizer.adam),
    )

    optimizer = nnx.Optimizer(model, tx=optimizer, wrt=nnx.Param)

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    x, y = create_shakespeare(
        tokenizer, train_cfg.data_path, train_cfg.batch_size, train_cfg.seq_len
    )

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataset=(x, y),
        eval_dataset=(x, y),
        cfg=train_cfg,
        ckpt_mngr=ckpt_mngr,
    )


if __name__ == "__main__":
    main()
