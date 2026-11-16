from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import deltanet.config  # noqa: F401
from deltanet.losses import language_model_loss
from deltanet.trainer import train

CONFIG_DIR = str(Path(__file__).parent.parent / "configs")


def get_train_config(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR):
        cfg = compose("config", overrides=overrides)

    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize(
    "overrides",
    [
        ["training=short", "training.seq_len=32", "training.num_steps=5"],
        ["training=short", "training.seq_len=64", "training.num_steps=5"],
    ],
)
def test_train_job(make_deltanet, make_adam_optimizer, make_batch, overrides):
    cfg = get_train_config(overrides)

    model = make_deltanet(**cfg.model.model)
    optimizer = make_adam_optimizer(model, cfg.training.optimizer)

    full_batch = make_batch()
    x = full_batch[:, : cfg.training.seq_len]
    y = full_batch[:, 1 : cfg.training.seq_len + 1]

    train(
        model=model,
        optimizer=optimizer,
        loss_fn=language_model_loss,
        train_dataset=(x, y),
        eval_dataset=(x, y),
        cfg=cfg.training,
        ckpt_mngr=None,
    )
