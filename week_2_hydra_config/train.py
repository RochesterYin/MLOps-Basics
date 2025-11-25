import torch
import hydra
import wandb
import logging

import pandas as pd
import pytorch_lightning as pl
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data import DataModule
from model import ColaModel

logger = logging.getLogger(__name__)


class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        sentences = val_batch["sentence"]

        # Move inputs to the same device as the model (MPS/GPU/CPU)
        device = pl_module.device
        input_ids = val_batch["input_ids"].to(device)
        attention_mask = val_batch["attention_mask"].to(device)
        labels = val_batch["label"].to(device)

        outputs = pl_module(input_ids, attention_mask)
        preds = torch.argmax(outputs.logits, 1)

        # Move to CPU for numpy conversion (required for MPS)
        labels_cpu = labels.cpu()
        preds_cpu = preds.cpu()

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels_cpu.numpy(), "Predicted": preds_cpu.numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")
    cola_data = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    cola_model = ColaModel(cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        monitor="valid/loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    # W&B logger - set mode to "offline" if not logged in
    import os
    wandb_mode = os.getenv("WANDB_MODE", "online")
    # Use environment variable for entity, or default to None (uses logged-in user)
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    wandb_logger = WandbLogger(
        project="MLOps Basics", 
        entity=wandb_entity,  # None means use logged-in user
        mode=wandb_mode
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplesVisualisationLogger(cola_data), early_stopping_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        accelerator="auto",  # Automatically detects MPS, GPU, or CPU
    )
    trainer.fit(cola_model, cola_data)
    wandb.finish()


if __name__ == "__main__":
    main()
