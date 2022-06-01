"""
Changelog:
==========

0.0.1:
* First implementation of the Multi-Objective CNN Benchmark.
"""
import logging
import random
import time
from typing import Union, Dict, List, Tuple, Any

import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
import tqdm
from ConfigSpace.conditions import GreaterThanCondition
from torch.utils.data import TensorDataset, DataLoader

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from hpobench.util.data_manager import CNNDataManager

__version__ = '0.0.1'

logger = logging.getLogger('MO_CNN')


class AccuracyTop1:

    def __init__(self):
        self.reset()

        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.sum = 0
        self.cnt = 0

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        self.sum += y_pred.topk(1)[1].eq(y_true.argmax(-1).reshape(-1, 1).expand(-1, 1)).float().sum().to('cpu').numpy()
        self.cnt += y_pred.size(0)
        return self.sum / self.cnt


class Net(nn.Module):
    """
    The model to optimize
    """

    def __init__(self, config: Dict, input_shape: Tuple = (3, 28, 28),
                 num_classes: Union[int, None] = 10):
        super(Net, self).__init__()
        inp_ch = input_shape[0]
        layers = []
        for i in range(config['n_conv_layers']):
            out_ch = config['conv_layer_{}'.format(i)]
            ks = config['kernel_size']
            layers.append(nn.Conv2d(inp_ch, out_ch, kernel_size=ks, padding=(ks - 1) // 2))
            layers.append(nn.ReLU())
            if config['batch_norm']:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            inp_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1) if config['global_avg_pooling'] else nn.Identity()
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()

        inp_n = self._get_conv_output(input_shape)

        layers = [nn.Flatten()]
        for i in range(config['n_fc_layers']):
            out_n = config['fc_layer_{}'.format(i)]

            layers.append(nn.Linear(inp_n, out_n))
            layers.append(nn.ReLU())

            inp_n = out_n

        layers.append(nn.Linear(inp_n, num_classes))
        self.fc_layers = nn.Sequential(*layers)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape: Tuple) -> int:
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        output_feat = self.pooling(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = self.fc_layers(x)
        return x

    def train_fn(self, optimizer: torch.optim.Optimizer, criterion: Any, loader: DataLoader, device: torch.device):
        """
        Training method

        Parameters
        ----------
        optimizer
            optimization algorithm
        criterion
            loss function
        loader
            data loader for either training or testing set
        device
            Either CPU or GPU
        Returns
        -------
        accuracy on the data
        """
        accuracy = AccuracyTop1()
        self.train()

        acc = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)

            loss = criterion(logits, labels.argmax(-1))
            loss.backward()
            optimizer.step()

            acc = accuracy(labels, logits)

        return acc

    def eval_fn(self, loader: DataLoader, device: torch.device):
        """
        Evaluation method

        Parameters
        ----------
        loader:
            data loader for either training or testing set
        device:
            torch device

        Returns
        -------
        accuracy on the data
        """
        accuracy = AccuracyTop1()
        self.eval()

        acc = 0
        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc = accuracy(labels, outputs)

        return acc


class CNNBenchmark(AbstractMultiObjectiveBenchmark):
    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """
        Parameters
        ----------
        dataset : str
            One of fashion, flower.
        rng : np.random.RandomState, int, None
            Random seed for the benchmark's random state.
        """

        super(CNNBenchmark, self).__init__(rng=rng)
        allowed_datasets = ["fashion", "flower"]
        assert dataset in allowed_datasets, f'Requested data set is not supported. Must be one of ' \
                                            f'{", ".join(allowed_datasets)}, but was {dataset}'
        logger.info(f'Start Benchmark on dataset {dataset}')

        self.dataset = dataset
        self.__seed_everything()

        # Dataset loading
        data_manager = CNNDataManager(dataset=self.dataset)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = data_manager.load()

        self.output_classes = self.y_train.shape[1]
        self.input_shape = self.X_train.shape[1:4]

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all parameters for
        the CNN model.

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter('n_conv_layers', default_value=3, lower=1, upper=3, log=False),
            CS.UniformIntegerHyperparameter('n_fc_layers', default_value=3, lower=1, upper=3, log=False),
            CS.UniformIntegerHyperparameter('conv_layer_0', default_value=128, lower=16, upper=1024, log=True),
            CS.UniformIntegerHyperparameter('conv_layer_1', default_value=128, lower=16, upper=1024, log=True),
            CS.UniformIntegerHyperparameter('conv_layer_2', default_value=128, lower=16, upper=1024, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_0', default_value=32, lower=2, upper=512, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_1', default_value=32, lower=2, upper=512, log=True),
            CS.UniformIntegerHyperparameter('fc_layer_2', default_value=32, lower=2, upper=512, log=True),

            CS.UniformIntegerHyperparameter('batch_size', lower=1, upper=512, default_value=128, log=True),
            CS.UniformFloatHyperparameter('learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True),
            CS.CategoricalHyperparameter('batch_norm', default_value=False, choices=[False, True]),
            CS.CategoricalHyperparameter('global_avg_pooling', default_value=True, choices=[False, True]),
            CS.CategoricalHyperparameter('kernel_size', default_value=5, choices=[7, 5, 3])
        ])

        cs.add_conditions([
            # Add the conv_layer_1 (2nd layer) if we allow more than 1 (>1) `n_conv_layers`, and so on...
            GreaterThanCondition(cs.get_hyperparameter('conv_layer_1'), cs.get_hyperparameter('n_conv_layers'), 1),
            GreaterThanCondition(cs.get_hyperparameter('conv_layer_2'), cs.get_hyperparameter('n_conv_layers'), 2),
            GreaterThanCondition(cs.get_hyperparameter('fc_layer_1'), cs.get_hyperparameter('n_fc_layers'), 1),
            GreaterThanCondition(cs.get_hyperparameter('fc_layer_2'), cs.get_hyperparameter('n_fc_layers'), 2),
        ])

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters

        Fidelities
        ----------
        budget: int - [1, 25]
            Number of epochs to train

        Parameters
        ----------
        seed : int, None
            Fixing the seed for the ConfigSpace.ConfigurationSpace

        Returns
        -------
        ConfigSpace.ConfigurationSpace
        """
        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('budget', lower=1, upper=25, default_value=25, log=False)
        ])
        return fidelity_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Bag of baselines for multi-objective joint neural architecture search and '
                    'hyperparameter optimization',
            'references': ['@article{guerrero2021bag,'
                           'title   = {Bag of baselines for multi - objective joint neural architecture search and '
                           'hyperparameter optimization},'
                           'author  = {Guerrero-Viu, Julia and Hauns, Sven and Izquierdo, Sergio and Miotto, '
                           'Guilherme and Schrodi, Simon and Biedenkapp, Andre and Elsken, Thomas and Deng, '
                           'Difan and Lindauer, Marius and Hutter, Frank},},'
                           'journal = {arXiv preprint arXiv:2105.01015},'
                           'year    = {2021}}',
                           ],
            'code': 'https://github.com/automl/multi-obj-baselines',
        }

    @staticmethod
    def get_objective_names() -> List[str]:
        """Get the names of the objectives reported in the objective function."""
        return ['accuracy', 'model_size']

    def init_model(self, config: Union[CS.Configuration, Dict]) -> Net:
        """
        Function that returns the model initialized based on the configuration and fidelity
        """
        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        return Net(config, self.input_shape, self.output_classes)

    def __seed_everything(self):
        """Helperfunction: Make the benchmark deterministic by setting the correct seeds"""
        seed = self.rng.randint(0, 100000)
        logger.debug(f'Generate seed: {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _shuffle_data(self, rng=None, shuffle_valid=False) -> None:
        """
        Reshuffle the training data.

        Parameters
        ----------
        rng
            If 'rng' is None, the training idx are shuffled according to the class-random-state
        shuffle_valid: bool, None
            If true, shuffle the validation data. Defaults to False.
        """
        random_state = rng_helper.get_rng(rng, self.rng)

        train_idx = np.arange(len(self.X_train))
        random_state.shuffle(train_idx)
        self.X_train = self.X_train[train_idx]
        self.y_train = self.y_train[train_idx]

        if shuffle_valid:
            valid_idx = np.arange(len(self.X_valid))
            random_state.shuffle(valid_idx)
            self.X_valid = self.X_valid[valid_idx]
            self.y_valid = self.y_valid[valid_idx]

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           shuffle: bool = False,
                           **kwargs) -> Dict:
        """
        Train a CNN on either the flower or the fashion data set and return the performance on the validation
        data split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the CNN Model
        fidelity: Dict, CS.Configuration, None
            epoch: int - Values: [1, 50]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        shuffle: bool, None
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        kwargs

        Returns
        -------
        Dict -
            function_value : Dict
                negative_accuracy: float
                    1 - validation accuracy
                log_model_size: float
                    log10 of the number of parameters
            cost : time to train the network
            info : Dict
                train_accuracy : float,
                training_cost : float,
                valid_accuracy : float,
                valid_cost : float,
                test_accuracy : float,
                test_cost : float,
                model_size : int,
                fidelity : Dict
                    used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        self.__seed_everything()

        if shuffle:
            self._shuffle_data(rng=self.rng, shuffle_valid=False)

        time_in = time.time()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f'We use the device: {device}')

        # initializing model
        model = self.init_model(configuration).to(device)
        epochs = fidelity['budget']

        optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate_init'])
        criterion = torch.nn.CrossEntropyLoss()

        ds_train = TensorDataset(self.X_train, self.y_train)
        ds_train = DataLoader(ds_train, batch_size=configuration['batch_size'], shuffle=True)

        ds_val = TensorDataset(self.X_valid, self.y_valid)
        ds_val = DataLoader(ds_val, batch_size=configuration['batch_size'], shuffle=True)

        ds_test = TensorDataset(self.X_test, self.y_test)
        ds_test = DataLoader(ds_test, batch_size=configuration['batch_size'], shuffle=True)

        start = time.time()
        t = tqdm.tqdm(total=epochs)

        train_accuracy = 0
        for epoch in range(epochs):
            train_accuracy = model.train_fn(optimizer, criterion, ds_train, device).item()
            t.set_postfix(train_accuracy=train_accuracy)
            t.update()
        training_runtime = time.time() - start

        num_params = np.sum([p.numel() for p in model.parameters()]).item()
        start = time.time()
        val_accuracy = model.eval_fn(ds_val, device).item()
        eval_valid_runtime = time.time() - start
        start = time.time()
        test_accuracy = model.eval_fn(ds_test, device).item()
        eval_test_runtime = time.time() - start

        t.set_postfix(
            train_acc=train_accuracy,
            val_acc=val_accuracy,
            tst_acc=test_accuracy,
            len=np.log10(num_params),
            train_runtime=training_runtime,
            eval_valid_runtime=eval_valid_runtime,
            eval_test_runtime=eval_test_runtime,
        )
        t.close()

        elapsed_time = time.time() - time_in

        return {'function_value': {'negative_accuracy': 1 - val_accuracy,
                                   'log_model_size': float(np.log10(num_params))},
                'cost': float(training_runtime),
                'info': {'train_accuracy': train_accuracy,
                         'training_cost': training_runtime,
                         'valid_accuracy': val_accuracy,
                         'valid_cost': eval_valid_runtime,
                         'test_accuracy': test_accuracy,
                         'test_cost': eval_test_runtime,
                         'total_time': elapsed_time,
                         'model_size': num_params,
                         'fidelity': fidelity}
                }

    @AbstractMultiObjectiveBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                shuffle: bool = False,
                                **kwargs) -> Dict:
        """
        Train a CNN on both the train adn validation split of either the flower or the fashion data set and
        get the test results.
        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the CNN Model
        fidelity: Dict, CS.Configuration, None
            epoch: int - Values: [1, 50]
                Number of epochs an architecture was trained.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.
            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        shuffle: bool, None
            If ``True``, shuffle the training idx. If no parameter ``rng`` is given, use the class random state.
            Defaults to ``False``.
        kwargs

        Returns
        -------
        Dict -
            function_value : Dict
                negative_accuracy: float
                    1 - test accuracy
                log_model_size: float
                    log10 of the number of parameters
            cost : time to train the network
            info : Dict
                train_accuracy : float,
                training_cost : float,
                test_accuracy : float,
                test_cost : float,
                model_size : int,
                fidelity : Dict
                    used fidelities in this evaluation
        """

        time_in = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        self.__seed_everything()

        if shuffle:
            self._shuffle_data(rng=self.rng, shuffle_valid=False)

        train_X = torch.vstack((self.X_train, self.X_valid))
        y_train = torch.cat((self.y_train, self.y_valid))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # initializing model
        model = self.init_model(configuration).to(device)
        epochs = fidelity['budget']

        optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate_init'])
        criterion = torch.nn.CrossEntropyLoss()

        ds_train = TensorDataset(train_X, y_train)
        ds_train = DataLoader(ds_train, batch_size=configuration['batch_size'], shuffle=True)

        ds_test = TensorDataset(self.X_test, self.y_test)
        ds_test = DataLoader(ds_test, batch_size=configuration['batch_size'], shuffle=True)

        start = time.time()
        t = tqdm.tqdm(total=epochs)

        train_accuracy = 0
        for epoch in range(epochs):
            train_accuracy = model.train_fn(optimizer, criterion, ds_train, device).item()
            t.set_postfix(train_accuracy=train_accuracy)
            t.update()
        training_runtime = time.time() - start

        num_params = np.sum([p.numel() for p in model.parameters()])
        start = time.time()
        test_accuracy = model.eval_fn(ds_test, device).item()
        eval_test_runtime = time.time() - start

        t.set_postfix(
            train_acc=train_accuracy,
            tst_acc=test_accuracy,
            len=np.log10(num_params),
            eval_train_runtime=training_runtime,
            eval_test_runtime=eval_test_runtime,

        )
        t.close()

        elapsed_time = time.time() - time_in

        return {'function_value': {'negative_accuracy': 1 - test_accuracy,
                                   'log_model_size': float(np.log10(num_params))},
                'cost': training_runtime,
                'info': {'train_accuracy': train_accuracy,
                         'training_cost': training_runtime,
                         'test_accuracy': test_accuracy,
                         'test_cost': eval_test_runtime,
                         'total_time': elapsed_time,
                         'model_size': num_params,
                         'fidelity': fidelity}
                }


class FashionCNNBenchmark(CNNBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(FashionCNNBenchmark, self).__init__(dataset='fashion', rng=rng, **kwargs)


class FlowerCNNBenchmark(CNNBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(FlowerCNNBenchmark, self).__init__(dataset='flower', rng=rng, **kwargs)


__all__ = ["FashionCNNBenchmark",
           "FlowerCNNBenchmark"]
