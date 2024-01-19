"""
Author: Charlie
Purpose: Autoencoder using lighting
"""


import os
import torch
import lightning as L

from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.neural.layers import Encoder, Decoder


class Network(L.LightningModule):
    """
    Create basic neural network using lightning
    """

    def __init__(self, params):
        """
        Define autoencoder parameters and architecture

        Parameters:
        - params: user defined YAML parameters
        """

        super().__init__()

        # Load: Model Parameters

        self.obj = params["network"]["objective"]
        self.opti = params["network"]["optimizer"]
        self.alpha = params["network"]["learning_rate"]
        self.num_epochs = params["network"]["num_epochs"]
        self.space_size = params["network"]["space_size"]
        self.sample_size = params["datasets"]["sample_size"]

        # Define: Model Architecture

        self.encoder = Encoder(self.sample_size)
        num_features = self.encoder.output_features

        self.downsample = torch.nn.Linear(num_features, self.space_size)
        self.upsample = torch.nn.Linear(self.space_size, num_features)

        self.decoder = Decoder(self.sample_size, num_features)

    def configure_optimizers(self):
        """
        Create DL learning rate optimizer and learning rate schedular
        """

        # Create: Optimzation Routine

        if self.opti == 0:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.alpha)
        elif self.opti == 1:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.alpha)
        else:
            raise NotImplementedError

        # Create: Learning Rate Schedular

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def objective(self, labels, preds, choice=0):
        """
        Invoke reconstruction objective function

        Parameters:
        - preds (torch.tensor[float]): model predictions
        - labels (torch.tensor[float]): human truth annotations

        Returns:
        - (torch.tensor[float]): reconstruction error
        """

        if choice == 0:
            obj = torch.nn.MSELoss()
        elif choice == 1:
            obj = torch.nn.BCELoss()
            labels += 1e-6
        else:
            raise NotImplementedError

        return obj(labels, preds)

    def training_step(self, batch, batch_idx):
        """
        Run iteration for training dataset

        Parameters:
        - batch (tuple[torch.tensor[float]]): dataset mini-batch
        - batch_idx (int): index of current mini-batch

        Returns:
        - (torch.tensor[float]): Mini-batch loss
        """

        samples, labels = batch
        batch_size = samples.size()[0]

        # Gather: Predictions

        _, _, preds = self(samples)

        # Calculate: Objective Loss

        loss = self.objective(samples, preds, choice=self.obj)

        self.log("train_error", loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Run iteration for validation dataset

        Logs objective loss and confusion matrix measures

        Parameters:
        - batch (tuple[torch.tensor[float]]): dataset mini-batch
        - batch_idx (int): index of current mini-batch

        Returns:
        - (torch.tensor[float]): Mini-batch loss
        """

        samples, labels = batch
        batch_size = samples.size()[0]

        # Gather: Predictions

        _, _, preds = self(samples)

        # Calculate: Objective Loss

        loss = self.objective(samples, preds, choice=0)
        self.log("valid_error", loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

    def forward(self, x):
        """
        Run DL forward pass for predictions

        Parameters:
        - x (torch.tensor[float]): data input as batch

        Returns:
        - (torch.tensor[float]): DL predictions
        """

        x = self.encoder(x)

        shape = x.size()
        e_x = x.view(x.size()[0], -1)

        m_x = self.downsample(e_x)

        x = self.upsample(m_x)
        x = x.view(shape)

        d_x = self.decoder(x)

        return e_x, m_x, d_x


def load_trained_model(params):

    folder = os.path.join(params["paths"]["discovery"], "checkpoints")
    path = os.path.join(folder, os.listdir(folder)[-1])

    model = Network.load_from_checkpoint(path, params=params)

    return model
