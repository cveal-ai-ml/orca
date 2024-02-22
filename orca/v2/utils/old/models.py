
import sys
import torch

from tqdm import tqdm
from torchvision import models

from torch.utils.data import DataLoader


class Network(torch.nn.Module):

    def __init__(self):

        super().__init__()

        if sys.platform == "darwin":
            device = "mps"
        else:
            device = "cuda"

        self.device = device
        self.weights = models.ResNet50_Weights.DEFAULT
        self.arch_full = models.resnet50(weights=self.weights)
        self.arch_full = self.arch_full.eval()
        self.arch_full = self.arch_full.to(self.device)

        self.arch_part = models.resnet50(weights=self.weights)
        self.arch_part = torch.nn.Sequential(*list(self.arch_part.children())[:-1])

        self.arch_part = self.arch_part.eval()
        self.arch_part = self.arch_part.to(self.device)

    def forward(self, x):

        return self.arch_full(x), self.arch_part(x)

    def test(self, dataset):

        dataset = DataLoader(dataset, shuffle=False, batch_size=256)

        all_preds, all_features = [], []
        for sample, _ in tqdm(dataset, desc="Processing"):
            sample = sample.to(self.device)

            with torch.no_grad():
                preds, features = self(sample)
                preds = preds.softmax(dim=1)
                features = features.squeeze()

            all_preds.append(preds.cpu())
            all_features.append(features.cpu())

        all_preds = torch.vstack(all_preds).numpy()
        all_features = torch.vstack(all_features).numpy()

        return {"confidences": all_preds, "features": all_features}
