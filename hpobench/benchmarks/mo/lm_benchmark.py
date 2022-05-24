"""
Changelog:
==========

0.0.1:
* First implementation of the Multi-Objective Language Model Benchmark.
"""
from typing import Union, Dict, List
import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
import logging
import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractMultiObjectiveBenchmark
from hpobench.util.data_manager import LanguageModelDataManager
from hpobench.dependencies.lm.tokenize_util import batchify
from hpobench.dependencies.lm.model import TransformerModel
import time
import math
import tqdm
import random

__version__ = '0.0.1'

logger = logging.getLogger('LM_Bench')


class LanguageModelBenchmark(AbstractMultiObjectiveBenchmark):
    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """
        Tranformer based multi-objective language model benchmark

        Parameters
        ----------
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks

        Transformer Model is based on : "https://arxiv.org/pdf/1706.03762.pdf"
        """
        super(LanguageModelBenchmark, self).__init__(rng=rng)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        data_manager = LanguageModelDataManager(self.device)
        self.X_train, self.X_valid, self.X_test = data_manager.load()
        self.ntokens = len(data_manager.corpus.dictionary)
        self.__seed_everything()
        self.variable = {"eval_batch_size": 10,
                         "nlayers": 2,
                         "bptt": 35,
                         "tied": True,
                         # Number of attention head
                         "nhead": 2,
                         "ntoken": self.ntokens
                         }

    def __seed_everything(self):
        """Helperfunction: Make the benchmark deterministic by setting the correct seeds"""
        seed = self.rng.randint(0, 100000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'batch_size', default_value=128, lower=8, upper=256
            ),
            CS.UniformIntegerHyperparameter(
                'emsize', default_value=128, lower=32, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'lr_factor', default_value=50, lower=1, upper=100, log=True
            ),
            CS.UniformFloatHyperparameter(
                'lr', default_value=5, lower=1, upper=50, log=True
            ),
            CS.UniformFloatHyperparameter(
                'dropout', default_value=0.99, lower=0, upper=0.99
            ),
            CS.UniformFloatHyperparameter(
                'clip', default_value=0.99, lower=0.1, upper=2
            )

        ])
        return cs

    @staticmethod
    def get_objective_names() -> List[str]:
        return ['log_perplexity', 'accuracy', 'time']

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:

        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'budget', lower=1, upper=81, default_value=81, log=False
            )
        ])
        return fidelity_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {
            'name': 'Multi-objective Asynchronous Successive Halving',
            'references': ['@article{schmucker2021multi,'
                           'title={Multi-objective Asynchronous Successive Halving},'
                           'author={Schmucker, Robin and Donini, Michele and Zafar, Muhammad Bilal and Salinas,'
                           ' David and Archambeau, C{\'e}dric},'
                           'journal={arXiv preprint arXiv:2106.12639},'
                           'year={2021}',
                           ],
        }

    def init_model(self, config: Union[CS.Configuration, Dict]):
        """ Function that returns the model initialized based on the configuration and fidelity
        """

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        model = TransformerModel(
            self.variable['ntoken'], config['emsize'], self.variable['nhead'], config['emsize'],
            self.variable['nlayers'], config['dropout'])

        return model

    @AbstractMultiObjectiveBenchmark.check_parameters
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
            epoch: int - Values: [1, 81]
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
                log_perplexity: float
            cost : time to train the network
            info : Dict
                validation_accuracy : float,
                test_accuracy : float,
                log_perplexity : float,
                negative_log_perplexity : float,
                training_cost : float,
                valid_cost : float,
                test_cost : float,
                fidelity : Dict
                    used fidelities in this evaluation
        """

        self.rng = rng_helper.get_rng(rng)
        self.__seed_everything()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ts_start = time.time()

        # batchify data
        batch_size = configuration['batch_size']
        train_data = batchify(self.X_train, batch_size=batch_size).to(device)
        val_data = batchify(self.X_valid, batch_size=self.variable["eval_batch_size"]).to(device)
        test_data = batchify(self.X_test, batch_size=self.variable["eval_batch_size"]).to(device)

        epochs = fidelity['budget']

        model = self.init_model(configuration).to(device)

        criterion = nn.CrossEntropyLoss()

        learning_rate = configuration['lr']
        learning_rate_factor = configuration['lr_factor']
        clip = configuration['clip']
        best_val_loss = None
        train_time = 0
        eval_time = 0

        t = tqdm.tqdm(total=epochs)
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = model.train_fun(self.ntokens, criterion, train_data, learning_rate, clip)
            train_time += time.time() - epoch_start_time
            start = time.time()
            val_loss, val_acc = model.eval_fun(self.ntokens, criterion, val_data)
            val_loss = np.clip(val_loss, 1e-10, 10)
            eval_time += start - time.time()

            t.set_postfix(val_accuracy=val_acc)
            t.update()

            # Taken from original experimental setup
            if not np.isfinite(val_loss):
                val_loss = 7

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                learning_rate /= learning_rate_factor

        start_time = time.time()
        _, test_acc = model.eval_fun(self.ntokens, criterion, test_data)
        eval_test_runtime = time.time() - start_time

        perplexity = math.exp(best_val_loss)
        log_perplexity = best_val_loss
        neg_log_perplexity = 10 - best_val_loss
        elapsed_time = ts_start - time.time()

        return {'function_value': {'log_perplexity': log_perplexity,
                                   'accuracy': val_acc.item(),
                                   'time': train_time + eval_time
                                   },
                'cost': elapsed_time,
                'info': {'train_accuracy': train_acc.item(),
                         'validation_accuracy': val_acc.item(),
                         'test_accuracy': test_acc.item(),
                         'log_perplexity': log_perplexity,
                         'perplexity': perplexity,
                         'negative_log_perplexity': neg_log_perplexity,
                         'training_cost': train_time,
                         'valid_cost': eval_time,
                         'test_cost': eval_test_runtime,
                         'fidelity': fidelity
                         }
                }

    @AbstractMultiObjectiveBenchmark.check_parameters
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
            epoch: int - Values: [1, 81]
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
                log_perplexity: float
            cost : time to train the network
            info : Dict
                validation_accuracy : float,
                test_accuracy : float,
                log_perplexity : float,
                negative_log_perplexity : float,
                training_cost : float,
                valid_cost : float,
                test_cost : float,
                fidelity : Dict
                    used fidelities in this evaluation
        """

        assert fidelity['epoch'] == 81, 'Only test data for the 81 epoch is available. '
        ts_start = time.time()

        self.rng = rng_helper.get_rng(rng)
        self.__seed_everything()

        # batchify data
        batch_size = configuration['batch_size']
        train_data = batchify(self.X_train, batch_size=batch_size)
        val_data = batchify(self.X_valid, batch_size=batch_size)
        train_data = np.vstack((train_data, val_data))
        train_data = torch.tensor(train_data).to(self.device)
        test_data = batchify(self.X_test, batch_size=self.variable["eval_batch_size"]).to(self.device)

        epochs = fidelity['budget']

        model = self.init_model(configuration).to(self.device)

        criterion = nn.CrossEntropyLoss()

        learning_rate = configuration['lr']
        learning_rate_factor = configuration['lr_factor']
        clip = configuration['clip']
        best_test_loss = None
        train_time = 0
        eval_time = 0
        t = tqdm.tqdm(total=epochs)
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_acc = model.train_fun(self.ntokens, criterion, train_data, learning_rate,
                                                    clip)
            train_time += time.time() - epoch_start_time
            start = time.time()

            test_loss, test_acc = model.eval_fun(self.ntokens, criterion, test_data)
            test_loss = np.clip(test_loss, 1e-10, 10)
            eval_time += time.time() - start

            t.set_postfix(test_accuracy=test_acc)
            t.update()
            if not np.isfinite(test_loss):
                test_loss = 7

            # Save the model if the validation loss is the best we've seen so far.
            if not best_test_loss or test_loss < best_test_loss:
                best_test_loss = test_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                learning_rate /= learning_rate_factor

        perplexity = math.exp(best_test_loss)
        log_perplexity = best_test_loss
        neg_log_perplexity = 10 - best_test_loss
        elapsed_time = ts_start - time.time()

        return {'function_value': {'log_perplexity': log_perplexity,
                                   'accuracy': test_acc.item(),
                                   'time': train_time + eval_time
                                   },
                'cost': elapsed_time,
                'info': {'train_accuracy': train_acc.item(),
                         'test_accuracy': test_acc.item(),
                         'log_perplexity': log_perplexity,
                         'perplexity': perplexity,
                         'negative_log_perplexity': neg_log_perplexity,
                         'training_cost': train_time,
                         'test_cost': eval_time,
                         'fidelity': fidelity
                         }
                }

    __all__ = ["LanguageModelBenchmark"]