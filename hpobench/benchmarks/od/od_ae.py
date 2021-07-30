import logging
import time
from typing import Union, Dict

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pytorch_lightning as pl

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.dependencies.od.backbones.mlp import MLP
from hpobench.dependencies.od.callbacks.checkpoint_saver import CheckpointSaver
from hpobench.dependencies.od.callbacks.earlystopping import EarlyStopping
from hpobench.dependencies.od.data_manager import OutlierDetectionDataManager
from hpobench.dependencies.od.models.autoencoder import Autoencoder
from hpobench.dependencies.od.utils.scaler import get_fitted_scaler

__version__ = '0.0.1'

logger = logging.getLogger('ODAutoencoder')


class ODAutoencoder(AbstractBenchmark):
    """
    A fully connected autoencoder implemented in PyTorch Lightning with a maximum of
    four layers (including latent layer) is optimized on the area under precission-recall curve (AUPR) metric.
    In addition to the neural architecture hyperparameters, the benchmark includes multiple regularization techniques,
    scalers, and activation functions to train the autoencoder on 15 different Outlier Detection
    DataSets (using a contamination ratio of 10%). Cross-validation, early stopping, and the number of training epochs
    as fidelity complete the benchmark.
    """

    def __init__(self,
                 dataset_name: str,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        Parameters
        ----------
        dataset_name : str
            Must be one of [
                "annthyroid", "arrhythmia", "breastw", "cardio", "ionosphere",
                "mammography", "musk", "optdigits", "pendigits", "pima",
                "satellite", "satimage-2", "thyroid", "vowels", "wbc"]
        rng : np.random.RandomState, int, None
        """
        self.rng = rng_helper.get_rng(rng)
        pl.seed_everything(self.rng.randint(0, 10000))

        # Load datamanager
        # It's important to call it before super
        # as AbstractBenchmark samples configuration space in which
        # the datamanager is needed
        self.dataset_name = dataset_name
        self.datamanager = OutlierDetectionDataManager(dataset_name, self.rng)

        super(ODAutoencoder, self).__init__(rng=self.rng)

    def get_features(self):
        """Returns the number of features for the given dataset name."""
        return self.datamanager.dataset.get_features()

    @AbstractBenchmark.check_parameters
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains the autoencoder with 4-fold cross-validation
        Training ends if AUPR is not getting better after 10 epochs.
        Returns one minus the mean of the best validation AUPRs.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the autoencoder
        fidelity: Dict, None
            Fidelity parameters for the autoencoder, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : 1 - mean of the best validation AUPRs from each split.
            cost : time to train and evaluate the model
            info : Dict
                train_loss : training loss (reconstruction loss)
                fidelity : used fidelities in this evaluation
        """
        start_time = time.time()

        train_losses = []
        val_auprs = []

        # Train four autoencoder
        for split in range(4):
            # Get data
            (X_train, y_train), (X_val, y_val) = self.datamanager.dataset.get_train_val_data(split=split)

            # Normalize data
            scaler = get_fitted_scaler(X_train, configuration["scaler"])
            if scaler is not None:
                X_train = scaler(X_train)
                X_val = scaler(X_val)

            # Set seed to ensure deterministic behaviour
            if rng:
                pl.seed_everything(rng)

            # Setup backbone + model
            backbone = MLP(self.datamanager.dataset.get_features(), configuration)
            model = Autoencoder(backbone, configuration)

            trainer = pl.Trainer(
                logger=[],
                checkpoint_callback=False,
                min_epochs=1,
                max_epochs=fidelity["epochs"],
                num_sanity_val_steps=0,
                check_val_every_n_epoch=1,
                deterministic=True,
                callbacks=[
                    EarlyStopping(activated=True, patience=10, worst_loss=0.0)
                ],
            )

            try:
                trainer.fit(
                    model,
                    train_dataloader=self.datamanager.dataset.get_loader(
                        X_train,
                        batch_size=configuration["batch_size"]),
                    val_dataloaders=self.datamanager.dataset.get_loader(
                        X_val,
                        y_val,
                        train=False)
                )

                # Get epoch with the highest validation aupr
                index = np.argmax(np.array(model.val_auprs))

                train_loss = model.train_losses[index]
                val_aupr = model.val_auprs[index]
            except Exception as e:
                logger.exception(e)
                train_loss = np.inf
                val_aupr = 0

            train_losses.append(train_loss)
            val_auprs.append(val_aupr)

        train_loss = float(np.mean(np.array(train_losses)))
        val_aupr = float(np.mean(np.array(val_auprs)))

        cost = time.time() - start_time

        return {
            'function_value': 1 - val_aupr,
            'cost': cost,
            'info': {
                'train_loss': train_loss,
                'val_aupr': val_aupr,
                'fidelity': fidelity
            }
        }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains the autoencoder with a given configuration on both the training
        and validation dataset. It is ensured that the combined dataset has the
        same contamination ratio as used in training. Finally, 1 - AUPR on
        the test dataset is reported. No early stopping is used.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the autoencoder
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : 1 - AUPR (on test dataset)
            cost : time to X_train and evaluate the model
            info : Dict
                train_valid_loss: Loss on the train+validation dataset
                fidelity : used fidelities in this evaluation
        """

        start_time = time.time()

        X_train, y_train = self.datamanager.dataset.get_train_data()
        X_test, y_test = self.datamanager.dataset.get_test_data()

        # Normalize data
        scaler = get_fitted_scaler(X_train, configuration["scaler"])
        if scaler is not None:
            X_train = scaler(X_train)
            X_test = scaler(X_test)

        # Setup backbone + model
        backbone = MLP(self.datamanager.dataset.get_features(), configuration)
        model = Autoencoder(backbone, configuration)

        trainer = pl.Trainer(
            logger=[],
            checkpoint_callback=False,
            # min epochs are recognized automatically based on current epoch
            min_epochs=1,
            max_epochs=int(fidelity["epochs"]),
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            deterministic=True,
            callbacks=[
                CheckpointSaver()
            ],
        )

        try:
            # We use the train+validation data to train and validate here
            trainer.fit(
                model,
                train_dataloader=self.datamanager.dataset.get_loader(
                    X_train,
                    y_train,
                    batch_size=configuration["batch_size"]),
                val_dataloaders=self.datamanager.dataset.get_loader(
                    X_train,
                    y_train,
                    train=False)
            )

            # Model is trained on the epoch with
            # the highest train+val AUPR (check checkpoint_saver for more information)
            trainer.test(
                model,
                self.datamanager.dataset.get_loader(
                    X_test,
                    y_test,
                    train=False),
                verbose=False
            )

            # Get index of best epoch
            index = np.argmax(np.array(model.val_auprs))

            # train and val AUPR should be roughtly the same here
            train_loss = float(model.train_losses[index])
            val_aupr = float(model.val_auprs[index])
            test_aupr = float(model.test_aupr)

        except Exception as e:
            logger.exception(e)
            train_loss = np.inf
            val_aupr = 0.0
            test_aupr = 0.0

        cost = time.time() - start_time

        return {
            'function_value': 1 - test_aupr,
            'cost': cost,
            'info': {
                'train_loss': train_loss,
                'val_aupr': val_aupr,
                'test_aupr': test_aupr,
                'fidelity': fidelity
            }
        }

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the autoencoder.

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """

        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)

        # Configuration space is depending on dataset features
        num_features = self.get_features()

        spacing = num_features / 4
        num_units = [num_features - spacing*i for i in range(5)]
        num_units = [1 if units < 1.0 else int(units) for units in num_units]  # Make sure we have at least one unit

        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=0, upper=3, default_value=2)
        num_units_layer_1 = CSH.UniformIntegerHyperparameter('num_units_layer_1',
                                                             lower=num_units[2], upper=num_units[0])
        num_units_layer_2 = CSH.UniformIntegerHyperparameter('num_units_layer_2',
                                                             lower=num_units[3], upper=num_units[1])
        num_units_layer_3 = CSH.UniformIntegerHyperparameter('num_units_layer_3',
                                                             lower=num_units[3], upper=num_units[1])
        num_latent_units = CSH.UniformIntegerHyperparameter('num_latent_units',
                                                            lower=num_units[4], upper=num_units[2])

        cs.add_hyperparameters([
            num_layers,
            num_units_layer_1,
            num_units_layer_2,
            num_units_layer_3,
            num_latent_units
        ])

        cs.add_condition(CS.GreaterThanCondition(num_units_layer_1, num_layers, 0))
        cs.add_condition(CS.GreaterThanCondition(num_units_layer_2, num_layers, 1))
        cs.add_condition(CS.GreaterThanCondition(num_units_layer_3, num_layers, 2))

        activation = CSH.CategoricalHyperparameter('activation', choices=['relu', 'swish', 'swish-1', 'tanh'],
                                                   default_value="relu")
        skip_connection = CSH.CategoricalHyperparameter('skip_connection', choices=[True, False],
                                                        default_value=False)
        batch_normalization = CSH.CategoricalHyperparameter('batch_normalization', choices=[True, False],
                                                            default_value=False)
        dropout = CSH.CategoricalHyperparameter('dropout', choices=[True, False], default_value=True)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', 0.2, 0.8, default_value=0.5)

        cs.add_hyperparameters([
            activation,
            skip_connection,
            batch_normalization,
            dropout,
            dropout_rate
        ])

        cs.add_condition(CS.EqualsCondition(dropout_rate, dropout, True))

        # Optimizer
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value=0.001, log=True)
        beta1 = CSH.UniformFloatHyperparameter('beta1', lower=0.85, upper=0.999, default_value=0.9)
        beta2 = CSH.UniformFloatHyperparameter('beta2', lower=0.9, upper=0.9999, default_value=0.999)
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=0.0, upper=0.1, default_value=0.01)

        cs.add_hyperparameters([
            lr,
            beta1,
            beta2,
            weight_decay
        ])

        # Batch size
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=512, default_value=128)
        cs.add_hyperparameter(batch_size)

        # Scaler
        scalers = ["None", "MinMax", "Standard"]
        choice = CSH.CategoricalHyperparameter(
            'scaler',
            scalers,
            default_value=scalers[0]
        )
        cs.add_hyperparameter(choice)

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the autoencoder.

        Fidelities
        ----------
        epochs: int - [10, 100]
            training epochs

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter("epochs", lower=10, upper=100, default_value=100, log=False),
        ])

        return fidel_space

    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        X_train, _ = self.datamanager.dataset.get_train_data()
        X_test, _ = self.datamanager.dataset.get_test_data()

        return {
            'name': 'OutlierDetection - AutoEncoder',
            'references': [
                '@misc{Rayana:2016 ,'
                'author = "Shebuti Rayana",'
                'year = "2016",'
                'title = “{ODDS} Library”,'
                'url = "http://odds.cs.stonybrook.edu",'
                'institution = "Stony Brook University, Department of Computer Sciences" }'

            ],
            'shape of train data': X_train.shape,
            'shape of test data': X_test.shape,
            'initial random seed': self.rng,
            'dataset_name': self.dataset_name,
            'contamination': self.datamanager.dataset.get_contamination_ratio()
        }


__all__ = [ODAutoencoder]
