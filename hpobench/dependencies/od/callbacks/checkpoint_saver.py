import pytorch_lightning as pl


class CheckpointSaver(pl.Callback):
    def __init__(self):
        self.best_checkpoint = None

    def save(self, trainer, model):
        """
        Saves the best weights locally.
        """

        checkpoint = {
            'should_stop': trainer.should_stop,
            'current_epoch': trainer.current_epoch,
            'weights': model.state_dict(),
        }

        self.best_checkpoint = checkpoint

    def load(self, trainer, model):
        checkpoint = None
        if hasattr(self, "best_checkpoint"):
            checkpoint = self.best_checkpoint

        if checkpoint is not None:
            trainer.should_stop = checkpoint["should_stop"]
            trainer.current_epoch = checkpoint["current_epoch"] + 1
            model.load_state_dict(checkpoint["weights"])

    def on_validation_end(self, trainer, model):
        # We already saved if the trainer has stopped
        if not trainer.should_stop:
            # Save if it's the best epoch
            if model.val_auprs[-1] == max(model.val_auprs):
                self.save(trainer, model)

    def on_test_start(self, trainer, model):
        # Load best weights here
        self.load(trainer, model)
