from pathlib import Path

import hydra
import optax
import orbax.checkpoint as ocp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from deltanet.losses import associative_recall_loss
from deltanet.model import DeltaNet
from deltanet.tasks.mqar import multiquery_ar
from deltanet.trainer import train


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    data_cfg = cfg.data
    model_cfg = cfg.model
    train_cfg = cfg.training

    common = {
        "vocab_size": data_cfg.vocab_size,
        "input_seq_len": data_cfg.seq_len,
        "num_kv_pairs": data_cfg.num_kv_pairs,
    }

    x_train, y_train = multiquery_ar(
        **common, num_examples=data_cfg.num_train_examples, seed=cfg.seed
    )
    x_test, y_test = multiquery_ar(
        **common, num_examples=data_cfg.num_test_examples, seed=cfg.seed + 1
    )

    rngs = nnx.Rngs(cfg.seed)
    model = DeltaNet(**model_cfg.model, vocab_size=data_cfg.vocab_size, rngs=rngs)
    optimizer = nnx.Optimizer(
        model, tx=optax.adamw(**train_cfg.optimizer), wrt=nnx.Param
    )

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=associative_recall_loss,
        train_dataset=(x_train, y_train),
        eval_dataset=(x_test, y_test),
        cfg=train_cfg,
        ckpt_mngr=ckpt_mngr,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()
