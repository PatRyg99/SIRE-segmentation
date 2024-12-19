import datetime
import os

import typer

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.sire.data.datamodule import AAASegmentationDataModule, AAATrackingDataModule
from src.sire.models.sire_seg import SIRESegmentation
from src.sire.models.sire_tracker import SIRETracker
from src.sire.utils.yaml import load_config

app = typer.Typer()


def init_model_dir(sub_dir: str):
    ct = datetime.datetime.now().strftime("%m-%d-%Y.%H-%M-%S")
    model_path = os.path.join(f".checkpoints/{sub_dir}/{ct}")
    os.makedirs(model_path, exist_ok=True)
    print("\nStoring checkpoints to:", model_path)

    return model_path


@app.command()
def train_segmentation(
    config_path: str = typer.Option("src/sire/configs/sire_lumen_segmentation_config.yml", "-c", "--config"),
    experiment_name: str = typer.Option("test-sigma=0.05", "-n", "--experiment-name"),
    wandb_offline: bool = typer.Option(False, "-w", "--wandb-offline"),
):
    model_path = init_model_dir("segmentation")
    config = load_config(config_path)

    wandb_logger = WandbLogger(
        project="SIRE-segmentation",
        entity="phd-rygiel",
        name=experiment_name,
        offline=wandb_offline,
        log_model=not wandb_offline,
    )
    checkpoint_callback = ModelCheckpoint(model_path, monitor="val/loss", mode="min", save_last=True)

    net = SIRESegmentation(**config["model"])
    datamodule = AAASegmentationDataModule(**config["datamodule"])

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )
    trainer.fit(net, datamodule=datamodule)


@app.command()
def train_tracker(
    config_path: str = typer.Option("src/sire/configs/sire_tracker_config.yml", "-c", "--config"),
    experiment_name: str = typer.Option("test", "-n", "--experiment-name"),
    wandb_offline: bool = typer.Option(False, "-w", "--wandb-offline"),
):
    model_path = init_model_dir("tracking")
    config = load_config(config_path)

    wandb_logger = WandbLogger(
        project="SIRE-tracking",
        entity="phd-rygiel",
        name=experiment_name,
        offline=wandb_offline,
        log_model=not wandb_offline,
    )
    checkpoint_callback = ModelCheckpoint(model_path, monitor="val/loss", mode="min", save_last=True)

    net = SIRETracker(**config["model"])
    datamodule = AAATrackingDataModule(**config["datamodule"])

    trainer = Trainer(
        **config["trainer"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )
    trainer.fit(net, datamodule=datamodule)


if __name__ == "__main__":
    app()
