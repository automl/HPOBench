import pytorch_lightning as pl


class EarlyStopping(pl.Callback):
    def __init__(self, activated: bool, patience: int, worst_loss: float):
        self.patience = patience
        self.lowest_loss = worst_loss
        self.counter = 0
        self.activated = activated

    def setup(self, trainer, model, stage):
        if not self.activated:
            trainer.should_stop = False

    def on_validation_end(self, trainer, model):
        if not self.activated:
            return

        last_loss = model.val_auprs[-1]

        if last_loss > self.lowest_loss:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
        else:
            self.lowest_loss = last_loss
            self.counter = 0
