from typing import Any, Optional
import torch
from lightning import LightningModule
from torchcfm import ConditionalFlowMatcher
from .components.augmentation import AugmentationModule
from .components.solver import FlowSolver


class CFMLitModule(LightningModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        flow_matcher: ConditionalFlowMatcher,
        scheduler: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.scheduler = scheduler
        self.criterion = torch.nn.MSELoss()

    def unpack_batch(self, batch):

        return batch

    def preprocess_batch(self, batch, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        X = self.unpack_batch(batch)
        # If no trajectory assume generate from standard normal
        x0 = X[0]
        x1 = X[1]
        return x0, x1

    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        x0, x1 = self.preprocess_batch(batch, training)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = self.net(t, xt, tr=True)
        return torch.nn.functional.mse_loss(vt, ut)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch, training=True)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        loss = self.step(batch)
        self.log(f"{prefix}/loss", loss)
        return {"loss": loss, "x": batch}

    def validation_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

