"""
How to use this benchmark:
--------------------------

We recommend using the containerized version of this benchmark.
If you want to use this benchmark locally (without running it via the corresponding container),
you need to perform the following steps.

Prerequisites:
==============
Conda environment in which the HPOBench is installed (pip install .). Activate your environment.
```
conda activate <Name_of_Conda_HPOBench_environment>
```

1. Clone from github:
=====================
Clone the learna tool and install it to your Conda environment.
```
git clone --single-branch --branch development https://github.com/PhMueller/learna.git
cd learna
pip install .
```

2. Install requirements:
========================
Install the necessary requirements defined in the environment.yml, which is in the downloaded
learna repository.
```
conda env update --name <Name_of_Conda_HPOBench_environment> --file ./environment.yml
```

3. Get the data:
================
Download the necessary files.

```
cd <path>/<to>/learna_repository

python -m python -m learna.data.download_and_build_eterna ./learna/data/secondaries_to_single_files.sh data/eterna data/eterna/interim/eterna.txt  # noqa 501
    && ./learna/data/download_and_build_rfam_taneda.sh \
    && ./learna/data/download_and_build_rfam_learn.sh \
    && mv data/rfam_learn/test data/rfam_learn_test \
    && mv data/rfam_learn/validation data/rfam_learn_validation \
    && mv data/rfam_learn/train data/rfam_learn_train \
    && rm -rf data/rfam_learn \
    && chmod -R 755 data/
```

Changelog:
==========
0.0.4
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.3:
* Standardize the structure of the meta information

0.0.1:
* First implementation
"""

import logging
import multiprocessing
import shutil
import tempfile
from pathlib import Path
from time import time
from typing import Dict, List, Union

import ConfigSpace as CS
import numpy as np
from learna.data.parse_dot_brackets import parse_dot_brackets
from learna.learna.agent import NetworkConfig, AgentConfig
from learna.learna.design_rna import design_rna
from learna.learna.environment import RnaDesignEnvironmentConfig
from learna.learna.learn_to_design_rna import learn_to_design_rna

import hpobench.config
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper

__version__ = '0.0.4'

logger = logging.getLogger('LearnaBenchmark')


class BaseLearna(AbstractBenchmark):
    def __init__(self, data_path: Union[str, Path], rng: Union[np.random.RandomState, int, None] = None):
        """
        Benchmark for the RNA learning task using RL proposed by Runge, Stoll, Falkner and Hutter.
        For further insights, we refer to the paper, which is cited in the meta information of this benchmark.

        Running each of the benchmarks takes up to two days.

        Parameters
        ----------
        data_path : str, Path
            Path to sequencing data, which is used in the experiments.
            To get the data, see Instructions '3. Get the data'
        rng : np.random.RandomState, int, None
            Random seed for the benchmarks
        """

        logger.warning('This Benchmark is not deterministic.')

        super(BaseLearna, self).__init__(rng=rng)
        self.train_sequences = parse_dot_brackets(dataset="rfam_learn_train",
                                                  data_dir=data_path,
                                                  target_structure_ids=range(1, 65000))

        self.validation_sequences = parse_dot_brackets(dataset="rfam_learn_validation",
                                                       data_dir=data_path,
                                                       target_structure_ids=range(1, 100))

        self.num_cores = 1

    def _validate(self, evaluation_timeout: Union[int, float], restore_path: Union[Path, str, None],
                  restart_timeout: Union[int, float], stop_learning: bool,
                  network_config: NetworkConfig, agent_config: AgentConfig, env_config: RnaDesignEnvironmentConfig) -> \
            Dict:
        """ Helper function to solve sequences. This procedure does not train a model. """
        evaluation_arguments = [[[validation_sequence],
                                 evaluation_timeout,  # timeout
                                 restore_path,
                                 stop_learning,
                                 restart_timeout,  # restart_timeout
                                 network_config,
                                 agent_config,
                                 env_config]
                                for validation_sequence in self.validation_sequences]

        with multiprocessing.Pool(self.num_cores) as pool:
            evaluation_results = pool.starmap(design_rna, evaluation_arguments)

        # evaluation_sequence_infos = {}
        evaluation_sum_of_min_distances = 0
        evaluation_sum_of_first_distances = 0
        evaluation_num_solved = 0

        for result in evaluation_results:
            # sequence_id = r[0].target_id
            result.sort(key=lambda e: e.time)

            # times = np.array(list(map(lambda e: e.time, r)))
            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, result)))

            evaluation_sum_of_min_distances += np.min(dists)
            evaluation_sum_of_first_distances += dists[0]

            evaluation_num_solved += np.min(dists) == 0.0

            # evaluation_sequence_infos[sequence_id] = {"num_episodes": len(r),
            #                                           "mean_time_per_episode": float((times[1:] - times[:-1]).mean()),
            #                                           "min_distance": float(dists.min()),
            #                                           "last_distance": float(dists[-1])}

        evaluation_info = {"num_solved": int(evaluation_num_solved),
                           "sum_of_min_distances": float(evaluation_sum_of_min_distances),
                           "sum_of_first_distances": float(evaluation_sum_of_first_distances),
                           # "sequence_infos": evaluation_sequence_infos
                           }

        return evaluation_info

    def _setup(self, configuration):
        """ Initialize the network, agent and environment. """
        config = self._fill_config(configuration)

        network_config = NetworkConfig(conv_sizes=[config["conv_size1"], config["conv_size2"]],
                                       conv_channels=[config["conv_channels1"], config["conv_channels2"]],
                                       num_fc_layers=config["num_fc_layers"],
                                       fc_units=config["fc_units"],
                                       num_lstm_layers=config["num_lstm_layers"],
                                       lstm_units=config["lstm_units"],
                                       embedding_size=config["embedding_size"])

        agent_config = AgentConfig(learning_rate=config["learning_rate"],
                                   batch_size=config["batch_size"],
                                   entropy_regularization=config["entropy_regularization"])

        env_config = RnaDesignEnvironmentConfig(reward_exponent=config["reward_exponent"],
                                                state_radius=config["state_radius"])

        return config, network_config, agent_config, env_config

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """ Interface for the objective function. """
        raise NotImplementedError()

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """ Interface for the test obejctive function. """
        raise NotImplementedError()

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'Learna',
                'references': ['@inproceedings{runge2019learning,'
                               'title     = {Learning to Design {RNA}},'
                               'author    = {Frederic Runge and Danny Stoll and Stefan Falkner and Frank Hutter},'
                               'booktitle = {International Conference on Learning Representations},'
                               'year      = {2019},}',
                               'https://ml.informatik.uni-freiburg.de/papers/19-ICLR-Learning-Design-RNA.pdf'],
                'code': 'https://github.com/automl/learna',
                'note': 'This benchmark is not deterministic, since tensorforce is not deterministic in this version.'
                }

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space for the Learna Benchmark.

        Parameters
        ----------
        seed : int, None
            Set the seed for the configuration space. Makes sampling from the configuration space deterministic.
        Returns
        -------
        CS.ConfigurationSpace
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)

        config_space = CS.ConfigurationSpace(seed=seed)
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=1e-3, log=True, default_value=5e-4),
            CS.UniformIntegerHyperparameter("batch_size", lower=32, upper=128, log=True, default_value=32),
            CS.UniformFloatHyperparameter("entropy_regularization",
                                          lower=1e-5, upper=1e-2, log=True, default_value=1.5e-3),
            CS.UniformFloatHyperparameter("reward_exponent", lower=1, upper=10, default_value=1),
            CS.UniformFloatHyperparameter("state_radius_relative", lower=0, upper=1, default_value=0),
            CS.UniformIntegerHyperparameter("conv_radius1", lower=0, upper=8, default_value=1),
            CS.UniformIntegerHyperparameter("conv_channels1", lower=1, upper=32, log=True, default_value=32),
            CS.UniformIntegerHyperparameter("conv_radius2", lower=0, upper=4, default_value=0),
            CS.UniformIntegerHyperparameter("conv_channels2", lower=1, upper=32, log=True, default_value=1),
            CS.UniformIntegerHyperparameter("num_fc_layers", lower=1, upper=2, default_value=2),
            CS.UniformIntegerHyperparameter("fc_units", lower=8, upper=64, log=True, default_value=50),
            CS.UniformIntegerHyperparameter("num_lstm_layers", lower=0, upper=2, default_value=0),
            CS.UniformIntegerHyperparameter("lstm_units", lower=1, upper=64, log=True, default_value=1),
            CS.UniformIntegerHyperparameter("embedding_size", lower=0, upper=4, default_value=1)
        ])
        return config_space

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        a Learna Benchmark.

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
            CS.UniformIntegerHyperparameter('cutoff_agent_per_sequence', lower=1, upper=600, default_value=600)
        ])

        return fidel_space

    @staticmethod
    def _fill_config(configuration: Dict) -> Dict:
        """ Helper function to fill the configuration space with the missing parameters """

        configuration["conv_size1"] = 1 + 2 * configuration["conv_radius1"]
        if configuration["conv_radius1"] == 0:
            configuration["conv_size1"] = 0
        del configuration["conv_radius1"]

        configuration["conv_size2"] = 1 + 2 * configuration["conv_radius2"]
        if configuration["conv_radius2"] == 0:
            configuration["conv_size2"] = 0
        del configuration["conv_radius2"]

        if configuration["conv_size1"] != 0:
            min_state_radius = configuration["conv_size1"] + configuration["conv_size1"] - 1
            max_state_radius = 32
            configuration["state_radius"] = int(min_state_radius
                                                + (max_state_radius - min_state_radius)
                                                * configuration["state_radius_relative"])
        else:
            min_state_radius = configuration["conv_size2"] + configuration["conv_size2"] - 1
            max_state_radius = 32
            configuration["state_radius"] = int(min_state_radius
                                                + (max_state_radius - min_state_radius)
                                                * configuration["state_radius_relative"])
        del configuration["state_radius_relative"]

        configuration["restart_timeout"] = None

        return configuration


class Learna(BaseLearna):

    def __init__(self, data_path: Union[str, Path], rng: Union[np.random.RandomState, int, None] = None):
        super(Learna, self).__init__(data_path=data_path, rng=rng)

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Start the learna experiment. Dont train a RL agent. Just optimize on the sequences.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
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
            function_value : sum of min distances
            cost : time to train and evaluate the model
            info : Dict
                num_solved : int - number of soved sequences
                sum_of_first_distances : metric describing quality of result
                fidelity : the used fidelities in this evaluation
        """

        self.rng = rng_helper.get_rng(rng, self_rng=self.rng)

        config, network_config, agent_config, env_config = self._setup(configuration)

        start_time = time()
        validation_info = self._validate(evaluation_timeout=fidelity["cutoff_agent_per_sequence"],
                                         restore_path=None,
                                         stop_learning=False,
                                         restart_timeout=config["restart_timeout"],
                                         network_config=network_config,
                                         agent_config=agent_config,
                                         env_config=env_config)
        cost = time() - start_time

        return {'function_value': validation_info["sum_of_min_distances"],
                'cost': cost,
                'info': {'num_solved': validation_info["num_solved"],
                         'sum_of_first_distances': validation_info["sum_of_first_distances"],
                         'fidelity': fidelity,
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Validate the Learna experiment.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
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
            function_value : sum of min distances
            cost : time to train and evaluate the model
            info : Dict
                num_solved : int - number of soved sequences
                sum_of_first_distances : metric describing quality of result
                fidelity : the used fidelities in this evaluation
        """
        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)


class MetaLearna(BaseLearna):
    def __init__(self, data_path: Union[str, Path], rng: Union[np.random.RandomState, int, None] = None):
        super(MetaLearna, self).__init__(data_path=data_path, rng=rng)
        self.config = hpobench.config.config_file

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Overwrites the BaseLearna fidelity space so that the default value of 'cutoff_agent_per_sequence' becomes 3600s.

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
            CS.UniformIntegerHyperparameter('cutoff_agent_per_sequence', lower=1, upper=3600, default_value=3600)
        ])

        return fidel_space

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        MetaLearna trains an RL agent for 1 hour on the given training set and then tries to solve the sequences.
        MetaLearna has only a maximum solving time per sequence of 60 seconds.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
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
            function_value : sum of min distances
            cost : time to train and evaluate the model
            info : Dict
                num_solved : int - number of solved sequences
                sum_of_first_distances : metric describing quality of result
                fidelity : the used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng, self.rng)
        tmp_dir = Path(tempfile.mkdtemp(dir=self.config.cache_dir))

        config, network_config, agent_config, env_config = self._setup(configuration)
        start_time = time()
        try:
            train_info = self._train(budget=fidelity["cutoff_agent_per_sequence"],
                                     tmp_dir=tmp_dir,
                                     network_config=network_config,
                                     agent_config=agent_config,
                                     env_config=env_config)

            validation_info = self._validate(evaluation_timeout=60,
                                             restore_path=tmp_dir,
                                             stop_learning=True,
                                             restart_timeout=config["restart_timeout"],
                                             network_config=network_config,
                                             agent_config=agent_config,
                                             env_config=env_config)

            cost = time() - start_time
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return {'function_value': validation_info["sum_of_min_distances"],
                'cost': cost,
                'info': {'train_info': train_info,
                         'validation_info': validation_info,
                         'fidelity': fidelity},
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[CS.Configuration, Dict, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Validate the MetaLearna experiment.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
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
            function_value : sum of min distances
            cost : time to train and evaluate the model
            info : Dict
                num_solved : int - number of solved sequences
                sum_of_first_distances : metric describing quality of result
                fidelity : the used fidelities in this evaluation
        """
        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)

    def _train(self, budget: Union[int, float], tmp_dir: Path, network_config: NetworkConfig,
               agent_config: AgentConfig, env_config: RnaDesignEnvironmentConfig) -> Dict:
        """
        Trains a RL agent.

        Parameters
        ----------
        budget : int, float
            Time in seconds to train a RL agent
        tmp_dir : Path
        network_config : NetworkConfig
        agent_config : AgentConfig
        env_config : RnaDesignEnvironmentConfig

        Returns
        -------
        Dict -
            num_solved : int - number of soved sequences
            sum_of_min_distances : metric describing quality of result
            sum_of_first_distances : metric describing quality of result
        """
        train_arguments = [self.train_sequences,
                           budget,  # timeout
                           self.num_cores,  # worker_count
                           tmp_dir,  # save_path
                           None,  # restore_path
                           network_config,
                           agent_config,
                           env_config]

        # need to run tensoflow in a separate thread otherwise the pool in _evaluate
        # does not work
        with multiprocessing.Pool(1) as pool:
            train_results = pool.apply(learn_to_design_rna, train_arguments)

        train_results = self._process_train_results(train_results)
        # train_sequence_infos = {}
        train_sum_of_min_distances = 0
        train_sum_of_last_distances = 0
        train_num_solved = 0

        for result in train_results.values():
            # sequence_id = r[0].target_id
            result.sort(key=lambda e: e.time)

            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, result)))

            train_sum_of_min_distances += np.min(dists)
            train_sum_of_last_distances += dists[-1]

            train_num_solved += np.min(dists) == 0.0

            # The train sequence info dict is too large
            # train_sequence_infos[sequence_id] = {"num_episodes": len(r),
            #                                      "min_distance": float(dists.min()),
            #                                      "last_distance": float(dists[-1])}

        train_info = {"num_solved": int(train_num_solved),
                      "sum_of_min_distances": float(train_sum_of_min_distances),
                      "sum_of_last_distances": float(train_sum_of_last_distances),
                      # "sequence_infos": train_sequence_infos
                      }

        return train_info

    @staticmethod
    def _process_train_results(train_results: List) -> Dict:
        """ Helper function to extract results into dictionary """
        results_by_sequence = {}
        for result in train_results:
            for seq in result:
                if seq.target_id not in results_by_sequence:
                    results_by_sequence[seq.target_id] = [seq]
                else:
                    results_by_sequence[seq.target_id].append(seq)

        return results_by_sequence
