import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F


class Autoencoder(pl.LightningModule):
    def __init__(self, backbone, config):
        super().__init__()

        self.config = config
        self.backbone = backbone

        self.train_losses = []
        self.val_auprs = []
        self.test_aupr = None

        # Setup latent layer
        self.latent_layer = nn.Linear(
            in_features=self.backbone.encoder_features[-1],
            out_features=self.backbone.latent_dim
        )

        # Setup output layer
        self.output_layer = nn.Linear(
            in_features=self.backbone.decoder_in_features[-2],
            out_features=self.backbone.decoder_features[-1]
        )

    def configure_optimizers(self):
        if self.config["optimizer"] == "AdamWOptimizer":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config["optimizer:AdamWOptimizer:lr"],
                betas=(self.config["optimizer:AdamWOptimizer:beta1"], self.config["optimizer:AdamWOptimizer:beta2"]),
                weight_decay=self.config["optimizer:AdamWOptimizer:weight_decay"]
            )
        elif self.config["optimizer"] == "SGDOptimizer":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config["optimizer:SGDOptimizer:lr"],
                momentum=self.config["optimizer:SGDOptimizer:momentum"],
                weight_decay=self.config["optimizer:SGDOptimizer:weight_decay"]
            )
        else:
            raise RuntimeError("Could not find the optimizer.")
            
        return optimizer

    @staticmethod
    def calculate_aupr(labels, scores):
        precision, recall, _ = precision_recall_curve(labels, scores)
        aupr = auc(recall, precision)

        return aupr
    
    @staticmethod
    def calculate_loss(x, x_hat):
        return F.mse_loss(x_hat, x)

    def forward(self, x):
        # Encode first
        encoder_outputs = self.backbone.encode(x)
        z = self.latent_layer(encoder_outputs[-1])

        # Decode
        x_hat = self.output_layer(self.backbone.decode(z, encoder_outputs))

        return x_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat)

        return loss

    def training_epoch_end(self, outputs):
        losses = torch.stack([o['loss'] for o in outputs]).cpu().numpy().flatten()
        self.train_losses.append(np.mean(losses))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat)

        return {
            'labels': y.flatten(),
            'loss': loss
        }

    def validation_epoch_end(self, outputs):
        labels = torch.stack([o['labels'] for o in outputs]).cpu().numpy().flatten()
        losses = torch.stack([o['loss'] for o in outputs]).cpu().numpy().flatten()

        aupr = self.calculate_aupr(labels, losses)
        self.val_auprs.append(aupr)

    def test_step(self, batch, _):
        x, y = batch
        x_hat = self(x)
        loss = self.calculate_loss(x, x_hat)

        return {
            'labels': y.flatten(),
            'loss': loss
        }

    def test_epoch_end(self, outputs):
        labels = np.array([o['labels'].item() for o in outputs]).flatten()
        losses = np.array([o['loss'].item() for o in outputs]).flatten()

        self.test_aupr = self.calculate_aupr(labels, losses)
