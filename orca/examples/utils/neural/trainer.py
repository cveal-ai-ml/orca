"""
Author: Charlie
Purpose: Deep Learning (DL) autoencoder using Lightning AI
"""


import lightning as L

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from utils.data import load_datasets
from utils.neural.models import Network


def run(params):
    """
    Train DL autoencoder using Lightning AI

    Parameters:
    - params (dict[str, any]): user defined parameters
    """

    # Load: Experiment Datasets

    datasets = load_datasets(params)

    # Create: DL Network

    model = Network(params)

    # Create: Logger

    exp_logger = CSVLogger(save_dir=params["paths"]["results"])

    # Create: Trainer

    strategy = params["system"]["strategy"]
    num_devices = params["system"]["num_devices"]
    accelerator = params["system"]["accelerator"]

    num_epochs = params["network"]["num_epochs"]

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(callbacks=[lr_monitor],
                        accelerator=accelerator, strategy=strategy,
                        devices=num_devices, max_epochs=num_epochs,
                        log_every_n_steps=1, logger=exp_logger)

    # Train: Model

    train, valid = datasets["train"], datasets["valid"]
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=valid)
