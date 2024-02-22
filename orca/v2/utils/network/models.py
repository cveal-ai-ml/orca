"""
Purpose: Pre-trained DL Network
Author: Charlie
"""


import torch
import torch.nn as nn

from tqdm import tqdm
from torchvision import models

from torch.utils.data import DataLoader


class Network(nn.Module):

    def __init__(self, device, batch_size):

        super().__init__()

        self.device = device
        self.batch_size = batch_size

        # Create: Network

        self.weights = models.ResNet50_Weights.DEFAULT
        self.arch = models.resnet50(weights=self.weights)

        self.encode = models.resnet50(weights=self.weights)
        self.encode = nn.Sequential(*list(self.encode.children())[:-1])

        # Configure: Network

        self.arch.eval()
        self.encode.eval()

        self.arch = self.arch.to(self.device)
        self.encode = self.encode.to(self.device)

    def forward(self, x):

        return self.arch(x), self.encode(x)

    def test(self, data):

        # Create: Pytorch DataLoader

        data = DataLoader(data, shuffle=False, batch_size=self.batch_size)

        # Gather: Model Predictions

        all_preds, all_features = [], []
        for sample, _ in tqdm(data, desc="NN Predictions"):

            sample = sample.to(self.device)

            with torch.no_grad():
                preds, features = self(sample)
                preds = preds.softmax(dim=1)
                features = features.squeeze()

            all_preds.append(preds.cpu())
            all_features.append(features.cpu())

        # Format: Model Predictions

        all_preds = torch.vstack(all_preds).numpy()
        all_features = torch.vstack(all_features).numpy()

        return {"confidences": all_preds, "features": all_features}
