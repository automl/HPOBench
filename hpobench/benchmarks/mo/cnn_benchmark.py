"""
Changelog:
==========

0.0.1:
* First implementation of the Multi-Objective CNN Benchmark.
"""
import pathlib
from typing import Union, Tuple, Dict
import ConfigSpace as CS
import numpy as np
import torch
import tqdm
import torch.nn as nn
import pandas as pd
import logging
from ConfigSpace.hyperparameters import Hyperparameter
import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import CNNDataManager
import time

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

    def __call__(self, y_true, y_pred):
        self.sum += y_pred.topk(1)[1].eq(y_true.argmax(-1).reshape(-1, 1).expand(-1, 1)).float().sum().to('cpu').numpy()
        self.cnt += y_pred.size(0)

        return self.sum / self.cnt


class Net(nn.Module):
    """
    The model to optimize
    """

    def __init__(self, config, input_shape=(3, 28, 28), num_classes=10):
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

        self.time_train = 0

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        output_feat = self.pooling(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = self.fc_layers(x)
        return x

    def train_fn(self, optimizer, criterion, loader, device):
        """
        Training method
        :param optimizer: optimization algorithm
        :param criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :return: accuracy on the data
        """
        accuracy = AccuracyTop1()
        self.train()

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Step
            optimizer.zero_grad()
            logits = self(images)

            loss = criterion(logits, labels.argmax(-1))
            loss.backward()
            optimizer.step()

            acc = accuracy(labels, logits)

        return acc

    def eval_fn(self, loader, device):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        accuracy = AccuracyTop1()
        self.eval()

        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc = accuracy(labels, outputs)

        return acc


class CNNBenchmark(AbstractBenchmark):
    """
    Parameters
        ----------
        dataset : str
            One of fashion, flower.
        rng : np.random.RandomState, int, None
            Random seed for the benchmark's random state.
    """

    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(CNNBenchmark, self).__init__(rng=rng)

        allowed_datasets = ["fashion", "flower"]
        assert dataset in allowed_datasets, f'Requested data set is not supported. Must be one of ' \
                                            f'{", ".join(allowed_datasets)}, but was {dataset}'
        logger.info(f'Start Benchmark on dataset {dataset}')
        
        self.dataset=dataset
        # Dataset loading

        data_manager = CNNDataManager(dataset=self.dataset)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = data_manager.load()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'n_conv_layers', default_value=3, lower=1, upper=3, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'conv_layer_0', default_value=128, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'conv_layer_1', default_value=128, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'conv_layer_2', default_value=128, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'n_fc_layers', default_value=3, lower=1, upper=3, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_0', default_value=32, lower=2, upper=512, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_1', default_value=32, lower=2, upper=512, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'fc_layer_2', default_value=32, lower=2, upper=512, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=1, upper=512, default_value=128, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=10 ** -5, upper=1, default_value=10 ** -3, log=True
            ),
            CS.CategoricalHyperparameter(
                'batch_norm', default_value=False, choices=[False, True]
            ),
            CS.CategoricalHyperparameter(
                'global_avg_pooling', default_value=True, choices=[False, True]
            ),
            CS.CategoricalHyperparameter(
                'kernel_size', default_value=5, choices=[7, 5, 3]
            )

        ])
        return cs

    @staticmethod
    def get_objectives():
        return ['accuracy', 'model_size']

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:

        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            CNNBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(iter_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        fidelity1 = dict(
            fixed=CS.Constant('budget', value=50),
            variable=CS.UniformIntegerHyperparameter(
                'budget', lower=1, upper=50, default_value=50, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        budget = fidelity1[iter_choice]
        subsample = fidelity2[subsample_choice]
        return budget, subsample

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Bag of baselines for multi-objective joint neural architecture search and hyperparameter optimization',
            'references': ['@article{guerrero2021bag,'
                           'title   = {Bag of baselines for multi - objective joint neural architecture search and hyperparameter optimization},'
                           'author  = {Guerrero-Viu, Julia and Hauns, Sven and Izquierdo, Sergio and Miotto, Guilherme and Schrodi, Simon and Biedenkapp, Andre and Elsken, Thomas and Deng, Difan and Lindauer, Marius and Hutter, Frank},},'
                           'journal = {arXiv preprint arXiv:2105.01015},'
                           'year    = {2021}}',
                           ],
            'code': 'https://github.com/automl/multi-obj-baselines',
        }

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = config.get_dictionary()
        return Net(config, (3, 16, 16), 17)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           shuffle: bool = False,
                           **kwargs) -> Dict:
        """

        Parameters
        ----------
        configuration
        fidelity: Dict, None
            epoch: int - Values: [1, 50]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)

            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.


        kwargs

        Returns
        -------
        Dict -
            function_value : Dict
                validation_accuracy: float
                model_size: float
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
        self.rng = rng_helper.get_rng(rng)
        print("fid",fidelity)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # initializing model
        model = self.init_model(configuration, fidelity, rng).to(device)
        epochs = fidelity['budget'] - 1

        optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate_init'])
        criterion = torch.nn.CrossEntropyLoss()
        
        self.X_train = torch.tensor(self.X_train).float()
        self.X_train = self.X_train.permute(0, 3, 1, 2)
        self.y_train = torch.tensor(self.y_train).long()
        

        ds_train = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        ds_train = torch.utils.data.DataLoader(ds_train, batch_size=configuration['batch_size'], shuffle=True)

        self.X_valid = torch.tensor(self.X_valid).float()
        self.X_valid = self.X_valid.permute(0, 3, 1, 2)
        self.y_valid = torch.tensor(self.y_valid).long()
        
        ds_val = torch.utils.data.TensorDataset(self.X_valid, self.y_valid)
        ds_val = torch.utils.data.DataLoader(ds_val, batch_size=configuration['batch_size'], shuffle=True)

        self.X_test = torch.tensor(self.X_test).float()
        self.X_test = self.X_test.permute(0, 3, 1, 2)
        self.y_test = torch.tensor(self.y_test).long()
        
        ds_test = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        ds_test = torch.utils.data.DataLoader(ds_test, batch_size=configuration['batch_size'], shuffle=True)

        start = time.time()
        t = tqdm.tqdm(total=epochs)
        for epoch in range(epochs):
            train_accuracy = model.train_fn(optimizer, criterion, ds_train, device)
            t.set_postfix(train_accuracy=train_accuracy)
            t.update()
        training_runtime = time.time() - start

        num_params = np.sum(p.numel() for p in model.parameters())
        start = time.time()
        val_accuracy = model.eval_fn(ds_val, device)
        eval_valid_runtime = time.time() - start
        start = time.time()
        test_accuracy = model.eval_fn(ds_test, device)
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

        return {'function_value': {'accuracy': val_accuracy,
                                   'model_size': num_params,
                                   },
                'cost': float(training_runtime + eval_valid_runtime),
                'info': {'train_accuracy': train_accuracy,
                         'training_cost': training_runtime,
                         'valid_accuracy': val_accuracy,
                         'valid_cost': eval_valid_runtime,
                         'test_accuracy': test_accuracy,
                         'test_cost': eval_test_runtime,
                         'model_size': num_params,
                         'fidelity': fidelity
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                shuffle: bool = False,
                                **kwargs) -> Dict:
        """
        Get the validated results. Runs a given configuration on the largest budget (here: 50).
        Parameters
        ----------
        configuration
        fidelity: Dict, None
            epoch: int - Values: [1, 50]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed. (Results after the first epoch: epoch = 1)

            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.

        kwargs

        Returns
        -------
        Dict -
            function_value : Dict
                validation_accuracy: float
                model_size: float
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

        # The result dict should contain already all necessary information -> Just swap the function value from valid
        # to test and the corresponding time cost
        assert fidelity['epoch'] == 50, 'Only test data for the 50. epoch is available. '

        self.rng = rng_helper.get_rng(rng)

        train_X = np.vstack((self.X_train, self.X_valid))
        self.y_train = pd.concat((self.y_train, self.y_valid))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # initializing model
        model = self.init_model(configuration, fidelity, rng).to(device)
        epochs = fidelity['budget'] - 1

        optimizer = torch.optim.Adam(model.parameters(), lr=configuration['learning_rate_init'])
        criterion = torch.nn.CrossEntropyLoss()

        train_X = torch.tensor(train_X).float()
        train_X = train_X.permute(0, 3, 1, 2)
        self.y_train = torch.tensor(self.y_train).long()
        

        self.X_test = torch.tensor(self.X_test).float()
        self.X_test = self.X_test.permute(0, 3, 1, 2)
        self.y_test = torch.tensor(self.y_test).long()
        
        

        ds_train = torch.utils.data.TensorDataset(train_X, self.y_train)
        ds_train = torch.utils.data.DataLoader(ds_train, batch_size=configuration['batch_size'], shuffle=True)

        ds_test = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        ds_test = torch.utils.data.DataLoader(ds_test, batch_size=configuration['batch_size'], shuffle=True)

        start = time.time()
        t = tqdm.tqdm(total=epochs)
        for epoch in range(epochs):
            train_accuracy = model.train_fn(optimizer, criterion, ds_train, device)
            t.set_postfix(train_accuracy=train_accuracy)
            t.update()
        training_runtime = time.time() - start

        num_params = np.sum(p.numel() for p in model.parameters())
        start = time.time()
        test_accuracy = model.eval_fn(ds_test, device)
        eval_test_runtime = time.time() - start

        t.set_postfix(
            train_acc=train_accuracy,
            tst_acc=test_accuracy,
            len=np.log10(num_params),
            eval_train_runtime=training_runtime,
            eval_test_runtime=eval_test_runtime,

        )
        t.close()

        return {'function_value': {'accuracy': test_accuracy,
                                   'model_size': num_params,
                                   },
                'cost': float(training_runtime + eval_test_runtime),
                'info': {'train_accuracy': train_accuracy,
                         'training_cost': training_runtime,
                         'test_accuracy': test_accuracy,
                         'test_cost': eval_test_runtime,
                         'model_size': num_params,
                         'fidelity': fidelity
                         }
                }


class FashionCNNBenchmark(CNNBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(FashionCNNBenchmark, self).__init__(dataset='fashion', rng=rng, **kwargs)


class FlowerCNNBenchmark(CNNBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(FlowerCNNBenchmark, self).__init__(dataset='flower', rng=rng, **kwargs)


__all__ = ["FashionCNNBenchmark",
           "FlowerCNNBenchmark"]
