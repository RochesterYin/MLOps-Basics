import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HUGGINGFACE_CO_URL", "https://huggingface.co")
from transformers import AutoTokenizer

from data import DataModule
from model import ColaModel


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    # 检测可用的设备（支持M3芯片的MPS）
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
    
    print(f"使用设备: {accelerator}")
    
    trainer = pl.Trainer(
        default_root_dir="logs",
        accelerator=accelerator,
        devices=devices,
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(cola_model, cola_data)


if __name__ == "__main__":
    main()
