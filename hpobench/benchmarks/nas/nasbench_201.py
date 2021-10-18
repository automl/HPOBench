"""
Interface to Benchmarks with Nas-Bench 201

https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md

How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.


1. Clone and install
====================
Since the data is downloaded automatically, you dont have to do anything but installing the hpobench.

Recommend: ``Python >= 3.6.0``

```
cd /path/to/HPOBench
pip install .
```

For more info about the nasbench201, please have a look at
https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NAS-Bench-201.md

Changelog:
==========
0.0.5
* Add for each benchmark a new one with a different fidelity space.
  The new fidelity space corresponds to the fidelity space in the DEHB paper.

0.0.4
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.3:
* Standardize the structure of the meta information

0.0.2:
* Use the new data format. The authors have evaluated each configuration on 3 different seeds.
  The objective function supports now to specify a seed. Possible values are 777, 888, 999, None.
  Explicitly setting the value to None means drawing a random seed.

0.0.1:
* First implementation
"""
import logging
from typing import Union, Dict, List, Text, Tuple
from copy import deepcopy

import ConfigSpace as CS
import numpy as np

import hpobench.util.rng_helper as rng_helper
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.data_manager import NASBench_201Data

__version__ = '0.0.5'
MAX_NODES = 4

logger = logging.getLogger('NASBENCH201')


class NasBench201BaseBenchmark(AbstractBenchmark):
    def __init__(self, dataset: str,
                 rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        """
        Benchmark interface to the NASBench201 Benchmarks. The NASBench201 contains
        results for architectures on 4 different data sets.

        We have split the "api" file from NASBench201 in separate files per data set.
        The original "api" file contains all data sets, but loading this single file took too much RAM.

        We recommend to not call this base class directly but using the correct subclass below.

        The parameter ``dataset`` indicates which data set was used for training.

        For each data set the metrics
        'train_acc1es', 'train_losses', 'train_times', 'eval_acc1es', 'eval_times', 'eval_losses' are available.
        However, the data sets report them on different data splits (train, train + valid, test, valid or test+valid).

        We summarize all information about the data sets in the following tables.

        Datastet        Metric      Avail.Epochs    Explanation             returned by HPOBENCH
        ----------------------------------------------------------------------------------------
        cifar10-valid   train       [0-199]         training set
        cifar10-valid   x-valid     [0-199]         validation set          objective function
        cifar10-valid   x-test
        cifar10-valid   ori-test    199             test set                objective function test

        cifar100        train       [0-199]         training set
        cifar100        x-valid     199             validation set
        cifar100        x-test      199             test set                objective function test
        cifar100        ori-test    [0-199]         validation + test set   objective function

        ImageNet16-120  train       [0-199]         training set
        ImageNet16-120  x-valid     199             validation set
        ImageNet16-120  x-test      199             test set                objective function test
        ImageNet16-120  ori-test    [0-199]         validation + test set   objective function


        We have also extracted the incumbents per split. We report the incumbent accuracy and loss performance
        i) by taking the maximum value across all seeds and configurations
        ii) averaged across the three available seeds

                                    i) The best possible incumbents (NO AVG!)                       ii) The "average" incumbent
        Datastet        Metric      (Index of Arch, Accuracy)       (Index, Loss)                   (Index of Arch, Accuracy)       (Index, Loss)
        ----------------------------------------------------------------------------------------------------------------------------------------------------------
        cifar10-valid   train       (258, 100.0)                    (2778, 0.001179278278425336)    (10154, 100)                    (2778, 0.0013082386429297428)
        cifar10-valid   x-valid     (6111, 91.71999999023437)       (14443, 0.3837750501537323)     (6111, 91.60666665039064)       (3888, 0.3894046771335602)
        cifar10-valid   x-test
        cifar10-valid   ori-test    (14174, 91.65)                  (3385, 0.3850496160507202)      (1459, 91.52333333333333)       (3385, 0.3995230517864227)

        cifar100        train       (9930, 99.948)                  (9930, 0.012630240231156348)    (9930, 99.93733333333334)       (9930, 0.012843489621082942)
        cifar100        x-valid     (13714, 73.71999998779297)      (13934, 1.1490126512527465)     (9930, 73.4933333577474)        (7361, 1.1600867895126343)
        cifar100        x-test      (1459, 74.28000004882813)       (15383, 1.1427113876342774)     (9930, 73.51333332112631)       (7337, 1.1747569534301758)
        cifar100        ori-test    (9930, 73.88)                   (13706, 1.1610547459602356)     (9930, 73.50333333333333)       (7361, 1.1696554500579834)

        ImageNet16-120  train       (9930, 73.2524719841793)        (9930, 0.9490517352046979)      (9930, 73.22918040138735)       (9930, 0.9524298415108582)
        ImageNet16-120  x-valid     (13778, 47.39999985758463)      (10721, 2.0826991437276203)     (10676, 46.73333327229818)      (10721, 2.0915397168795264)
        ImageNet16-120  x-test      (857, 48.03333317057292)        (12887, 2.0940088628133138)     (857, 47.31111100599501)        (11882, 2.106453532218933)
        ImageNet16-120  ori-test    (857, 47.083333353678384)       (11882, 2.0950548852284747)     (857, 46.8444444647895)         (11882, 2.1028235816955565)


        Note:
        - The parameter epoch is 0 indexed!
        - In the original data, the training splits are always marked with the key 'train' but they use different
          identifiers to refer to the available evaluation splits. We report them also in the table below.
        - We exclude the data set cifar10 from this benchmark.

         Some further remarks:
        - cifar10-valid is trained on the train split and tested on the validation split.
        - The train metrics are dictionaries with epochs (e.g. 0, 1, 2) as key and the metric as value.
          The evaluation metrics, however, have as key the identifiers, e.g. ori-test@0, with 0 indicating the epoch.
          Also, each data set reports values for all 200 epochs for a metric on the specified split
          and a single value on the 200th epoch for the other splits.

        Parameters
        ----------
        dataset : str
            One of cifar10-valid, cifar10, cifar100, ImageNet16-120.
        rng : np.random.RandomState, int, None
            Random seed for the benchmark's random state.
        """  # noqa: E501

        super(NasBench201BaseBenchmark, self).__init__(rng=rng)

        data_manager = NASBench_201Data(dataset=dataset)

        self.dataset = dataset
        self.data = data_manager.load()
        self.config_to_structure = NasBench201BaseBenchmark.config_to_structure_func(max_nodes=MAX_NODES)

    def dataset_mapping(self, dataset):
        mapping = {'cifar10-valid': ('x-valid', 'ori-test'),
                   'ImageNet16-120': ('ori-test', 'x-test'),
                   'cifar100': ('ori-test', 'x-test')}
        return mapping[dataset]

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           data_seed: Union[List, Tuple, int, None] = (777, 888, 999),
                           **kwargs) -> Dict:
        """
        Objective function for the NASBench201 benchmark.
        This functions sends a query to NASBench201 and evaluates the configuration.
        As already explained in the class definition, different data sets are trained on different splits.

        The table above gives a detailed summary over the available splits, epochs, and which identifier are used per
        dataset.

        Parameters
        ----------
        configuration
        fidelity: Dict, None
            epoch: int - Values: [1, 200]
                Number of epochs an architecture was trained.
                Note: the number of epoch is 1 indexed! (Results after the first epoch: epoch = 1)

            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        data_seed : List, Tuple, None, int
            The nasbench_201 benchmark include for each run 3 different seeds: 777, 888, 999.
            The user can specify which seed to use. If more than one seed is given, the results are averaged
            across the seeds but then the training time is the sum of the costs per seed.
            When this value is explicitly set to `None`, the function will chose randomly one out of [777, 888, 999].

            Note:
                For some architectures (configurations) no run was available. We've set missing values to an
                available value from another seed. Therefore, it is possible that run results are exactly the same for
                different seeds.

        kwargs

        Returns
        -------
        Dict -
            function_value : training precision
            cost : time to train the network
            info : Dict
                train_precision : float
                train_losses : float
                train_cost : float
                    Time needed to train the network for 'epoch' many epochs. If more than one seed is given,
                    this field is the sum of the training time per network
                eval_precision : float
                eval_losses : float
                eval_cost : float
                    Time needed to train the network for 'epoch many epochs plus the time to evaluate the network on the
                    evaluation split. If more than one seed is given, this field is the sum of the eval cost per network
                fidelity : Dict
                    used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng)

        if isinstance(data_seed, (List, Tuple)):
            assert len(data_seed) != 0, 'data_seed must not be empty'
            if len(set(data_seed)) != len(data_seed):
                logger.debug('There are some values more than once in the run_index. We remove the redundant entries.')
            data_seed = tuple(set(data_seed))
        elif isinstance(data_seed, int):
            data_seed = (data_seed, )
        elif data_seed is None:
            logger.debug('The data seed is explicitly set to None! A random seed will be selected.')
            data_seed = tuple(self.rng.choice((777, 888, 999), size=1))
        # Check if the data set seeds are valid
        else:
            raise ValueError(f'data seed has unknown data type {type(data_seed)}, '
                             f'but should be tuple or int (777,888,999)')

        assert len(set(data_seed) - {777, 888, 999}) == 0,\
            f'data seed can only contain the elements 777, 888, 999, but was {data_seed}'

        structure = self.config_to_structure(configuration)
        structure_str = structure.tostr()

        epoch = fidelity['epoch'] - 1
        data_seed = [str(seed) for seed in data_seed]
        valid_key, test_key = self.dataset_mapping(self.dataset)

        train_accuracies = [self.data[seed][structure_str]['train_acc1es'][f'{epoch}'] for seed in data_seed]
        train_losses = [self.data[seed][structure_str]['train_losses'][f'{epoch}'] for seed in data_seed]
        train_times = [np.sum((self.data[seed][structure_str]['train_times'][f'{e}']) for e in range(1, epoch + 1))
                       for seed in data_seed]

        valid_accuracies = [self.data[seed][structure_str]['eval_acc1es'][f'{valid_key}@{epoch}'] for seed in data_seed]
        valid_losses = [self.data[seed][structure_str]['eval_losses'][f'{valid_key}@{epoch}'] for seed in data_seed]
        valid_times = [np.sum((self.data[seed][structure_str]['eval_times'][f'{valid_key}@{e}'])
                              for e in range(1, epoch + 1)) for seed in data_seed]

        # There is a single value for the eval data per seed. (only epoch 200)
        test_accuracies = [self.data[seed][structure_str]['eval_acc1es'][f'{valid_key}@{199}'] for seed in data_seed]
        test_losses = [self.data[seed][structure_str]['eval_losses'][f'{valid_key}@{199}'] for seed in data_seed]
        test_times = [np.sum((self.data[seed][structure_str]['eval_times'][f'{test_key}@{199}'])
                             for e in range(1, epoch + 1)) for seed in data_seed]

        return {'function_value': float(100 - np.mean(valid_accuracies)),
                'cost': float(np.sum(valid_times) + np.sum(train_times)),
                'info': {'train_precision': float(100 - np.mean(train_accuracies)),
                         'train_losses': float(np.mean(train_losses)),
                         'train_cost': float(np.sum(train_times)),
                         'valid_precision': float(100 - np.mean(valid_accuracies)),
                         'valid_losses': float(np.mean(valid_losses)),
                         'valid_cost': float(np.sum(valid_times) + np.sum(train_times)),
                         'test_precision': float(100 - np.mean(test_accuracies)),
                         'test_losses': float(np.mean(test_losses)),
                         'test_cost': float(np.sum(train_times)) + float(np.sum(test_times)),
                         'fidelity': fidelity
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Get the validated results from the NASBench201. Runs a given configuration on the largest budget (here: 200).
        The test function uses all data set seeds (777, 888, 999).

        See also :py:meth:`~hpobench.benchmarks.nas.nasbench_201.objective_function`

        Parameters
        ----------
        configuration
        fidelity: Dict, None
            epoch: int - Values: [1, 200]
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
            function_value : evaluation precision
            cost : time to the network + time to validate
            info : Dict
                train_precision
                train_losses
                train_cost
                eval_precision
                eval_losses
                eval_cost
                fidelity : used fidelities in this evaluation
        """

        # The result dict should contain already all necessary information -> Just swap the function value from valid
        # to test and the corresponding time cost
        assert fidelity['epoch'] == 200, 'Only test data for the 200. epoch is available. '

        result = self.objective_function(configuration=configuration, fidelity=fidelity,
                                         data_seed=(777, 888, 999),
                                         rng=rng, **kwargs)
        result['function_value'] = result['info']['test_precision']
        result['cost'] = result['info']['test_cost']
        return result

    @staticmethod
    def config_to_structure_func(max_nodes: int):
        """
        From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
        Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        """
        def config_to_structure(config):
            genotypes = []
            for i in range(1, max_nodes):
                x_list = []
                for j in range(i):
                    node_str = f'{i}<-{j}'
                    op_name = config[node_str]
                    x_list.append((op_name, j))
                genotypes.append(tuple(x_list))
            return NasBench201BaseBenchmark._Structure(genotypes)
        return config_to_structure

    @staticmethod
    def get_search_spaces(xtype: str, name: str) -> List[Text]:
        """ obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
        From https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/models/__init__.py
        Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]
        """
        # pylint: disable=no-else-return
        if xtype == 'cell':
            NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
            SearchSpaceNames = {'nas-bench-201': NAS_BENCH_201}
            assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
            return SearchSpaceNames[name]
        else:
            raise ValueError('invalid search-space type is {:}'.format(xtype))

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Return the CS representation of the search space.
        From https://github.com/D-X-Y/AutoDL-Projects/blob/master/exps/algos/BOHB.py
        Author: https://github.com/D-X-Y [Xuanyi.Dong@student.uts.edu.au]

        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
        CS.ConfigurationSpace -
            Containing the benchmark's hyperparameter
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)

        search_space = NasBench201BaseBenchmark.get_search_spaces('cell', 'nas-bench-201')
        hps = [CS.CategoricalHyperparameter(f'{i}<-{j}', search_space) for i in range(1, MAX_NODES) for j in range(i)]
        cs.add_hyperparameters(hps)
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the NAS Benchmark 201.

        Fidelities:
         - epoch: int
         The loss / accuracy at `epoch`. Can be from 0 to 199.

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
            CS.UniformIntegerHyperparameter('epoch', lower=1, upper=200, default_value=200)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        return {'name': 'NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search',
                'references': ['@article{dong2020bench,'
                               'title   = {Nas-bench-201: Extending the scope of reproducible neural '
                               '           architecture search},'
                               'author  = {Dong, Xuanyi and Yang, Yi},'
                               'journal = {arXiv preprint arXiv:2001.00326},'
                               'year    = {2020}}',
                               'https://openreview.net/forum?id=HJxyZkBKDr',
                               ],
                'code': 'https://github.com/D-X-Y/AutoDL-Projects',
                }

    class _Structure:
        def __init__(self, genotype):
            assert isinstance(genotype, (list, tuple)), 'invalid class of genotype : {:}'.format(type(genotype))
            self.node_num = len(genotype) + 1
            self.nodes = []
            self.node_N = []
            for idx, node_info in enumerate(genotype):
                assert isinstance(node_info, (list, tuple)), 'invalid class of node_info : {:}'.format(type(node_info))
                assert len(node_info) >= 1, 'invalid length : {:}'.format(len(node_info))
                for node_in in node_info:
                    assert isinstance(node_in, (list, tuple)), 'invalid class of in-node : {:}'.format(type(node_in))
                    assert len(node_in) == 2 and node_in[1] <= idx, 'invalid in-node : {:}'.format(node_in)
                self.node_N.append(len(node_info))
                self.nodes.append(tuple(deepcopy(node_info)))

        def tostr(self):
            """ Helper function: Create a string representation of the configuration """
            strings = []
            for node_info in self.nodes:
                string = '|'.join([x[0] + '~{:}'.format(x[1]) for x in node_info])
                string = '|{:}|'.format(string)
                strings.append(string)
            return '+'.join(strings)

        def __repr__(self):
            return (
                '{name}({node_num} nodes with {node_info})'.format(name=self.__class__.__name__, node_info=self.tostr(),
                                                                   **self.__dict__))

        def __len__(self):
            return len(self.nodes) + 1

        def __getitem__(self, index):
            return self.nodes[index]


class Cifar10ValidNasBench201Benchmark(NasBench201BaseBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(Cifar10ValidNasBench201Benchmark, self).__init__(dataset='cifar10-valid', rng=rng, **kwargs)


class Cifar100NasBench201Benchmark(NasBench201BaseBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(Cifar100NasBench201Benchmark, self).__init__(dataset='cifar100', rng=rng, **kwargs)


class ImageNetNasBench201Benchmark(NasBench201BaseBenchmark):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(ImageNetNasBench201Benchmark, self).__init__(dataset='ImageNet16-120', rng=rng, **kwargs)


class _NasBench201BaseBenchmarkOriginal(NasBench201BaseBenchmark):

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        the NAS Benchmark 201.

        This fidelity space differs from the one above in its lower bound.
        The benchmark above enables the user to access the entire dataset, while this one reproduces the
        experiments from DEHB
        [DEHB](https://github.com/automl/DEHB/tree/937dd5cf48e79f6d587ea2ff408cb5ad9a8dce46/dehb/examples)

        Fidelities:
        epoch: int
            The loss / accuracy at `epoch`.

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

        # We use here the lower bound of 4 instead of 1.
        fidel_space.add_hyperparameters([
            CS.UniformIntegerHyperparameter('epoch', lower=12, upper=200, default_value=200)
        ])

        return fidel_space

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = NasBench201BaseBenchmark.get_meta_information()
        meta_information['note'] = \
            'This version of the benchmark implements the fidelity space defined in the DEHB paper.' \
            'See [DEHB](https://github.com/automl/DEHB/tree/937dd5cf48e79f6d587ea2ff408cb5ad9a8dce46/dehb/examples)'
        return meta_information


class Cifar10ValidNasBench201BenchmarkOriginal(_NasBench201BaseBenchmarkOriginal):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(Cifar10ValidNasBench201BenchmarkOriginal, self).__init__(dataset='cifar10-valid', rng=rng, **kwargs)


class Cifar100NasBench201BenchmarkOriginal(_NasBench201BaseBenchmarkOriginal):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(Cifar100NasBench201BenchmarkOriginal, self).__init__(dataset='cifar100', rng=rng, **kwargs)


class ImageNetNasBench201BenchmarkOriginal(_NasBench201BaseBenchmarkOriginal):

    def __init__(self, rng: Union[np.random.RandomState, int, None] = None, **kwargs):
        super(ImageNetNasBench201BenchmarkOriginal, self).__init__(dataset='ImageNet16-120', rng=rng, **kwargs)


__all__ = ["Cifar10ValidNasBench201Benchmark",
           "Cifar100NasBench201Benchmark",
           "ImageNetNasBench201Benchmark",
           "Cifar10ValidNasBench201BenchmarkOriginal",
           "Cifar100NasBench201BenchmarkOriginal",
           "ImageNetNasBench201BenchmarkOriginal"]
