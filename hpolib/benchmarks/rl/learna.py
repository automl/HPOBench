"""
How to use this benchmark:
--------------------------

Prerequisites:
- Conda environment in which the HPOlib is installed (pip install .)

1. Clone from github:
- Clone the learna tool via ```git clone https://github.com/automl/learna.git```

2. Install into HPOlib3 conda environment.
- ```conda env update --name <Name_of_Conda_HPOlib_environment> --file ./learna/environment.yml```

3. Activate the Environment which includes HPOlib3

4. Get the data
- cd <path>/<to>/learna
- python -m src.data.download_and_build_eterna ./src/data/secondaries_to_single_files.sh data/eterna data/eterna/interim/eterna.txt
- @./src/data/download_and_build_rfam_taneda.sh
- ./src/data/download_and_build_rfam_learn.sh && \
   mv data/rfam_learn/test data/rfam_learn_test && \
   mv data/rfam_learn/validation data/rfam_learn_validation && \
   mv data/rfam_learn/train data/rfam_learn_train && \
   rm -rf data/rfam_learn

5. Or use the containerized version. (Soon available)
"""

import sys
import multiprocessing

from pathlib import Path
from typing import Dict, Optional, List, Union
from time import time

import numpy as np
import ConfigSpace as CS

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper

# Import the learna package
# TODO: Make the learna package installable (?)
learna_path = list(Path.home().parent.rglob('learna'))[0]

from learna.data.parse_dot_brackets import parse_dot_brackets
from learna.learna.agent import NetworkConfig, AgentConfig
from learna.learna.environment import RnaDesignEnvironmentConfig
from learna.learna.design_rna import design_rna
from learna.learna.learn_to_design_rna import learn_to_design_rna

import hpolib.config


def _replace_multiple_char(str_to_modify: str, chars_to_replace: List, replace_by: Union[List, str]):
    if isinstance(replace_by, str):
        replace_by = [replace_by for i in range(len(chars_to_replace))]

    if len(replace_by) == 1:
        replace_by = [replace_by[0] for i in range(len(chars_to_replace))]

    for old_char, new_char in zip(chars_to_replace, replace_by):
        str_to_modify = str_to_modify.replace(old_char, new_char)
    return str_to_modify


class BaseLearna(AbstractBenchmark):
    def __init__(self, data_path: Union[str, Path]):
        """
        Benchmark for the RNA learning task using RL proposed by Runge, Stoll, Falkner and Hutter.
        For further insights, we refer to the paper, which is cited in the meta information of this benchmark.

        We set the time for each benchmark (Learna and MetaLearna) to 10 minutes per sequence.
        Thus, running each of the benchmarks may take two days.
        """
        super(BaseLearna, self).__init__()
        self.train_sequences = parse_dot_brackets(dataset="rfam_learn_train",
                                                  data_dir=data_path,
                                                  target_structure_ids=range(1, 65000))

        self.validation_sequences = parse_dot_brackets(dataset="rfam_learn_validation",
                                                       data_dir=data_path,
                                                       target_structure_ids=range(1, 101))

        self.num_cores = 1

    def _validate(self, evaluation_timeout: Union[int, float], restore_path: Union[Path, str, None],
                  restart_timeout: Union[int, float], stop_learning: bool,
                  network_config: NetworkConfig, agent_config: AgentConfig, env_config: RnaDesignEnvironmentConfig) -> \
            Dict:

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

        evaluation_sequence_infos = {}
        evaluation_sum_of_min_distances = 0
        evaluation_sum_of_first_distances = 0
        evaluation_num_solved = 0

        for r in evaluation_results:
            sequence_id = r[0].target_id
            r.sort(key=lambda e: e.time)

            times = np.array(list(map(lambda e: e.time, r)))
            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, r)))

            evaluation_sum_of_min_distances += dists.min()
            evaluation_sum_of_first_distances += dists[0]

            evaluation_num_solved += dists.min() == 0.0

            evaluation_sequence_infos[sequence_id] = {"num_episodes": len(r),
                                                      "mean_time_per_episode": float((times[1:] - times[:-1]).mean()),
                                                      "min_distance": float(dists.min()),
                                                      "last_distance": float(dists[-1])}

        evaluation_info = {"num_solved": int(evaluation_num_solved),
                           "sum_of_min_distances": float(evaluation_sum_of_min_distances),
                           "sum_of_first_distances": float(evaluation_sum_of_first_distances),
                           "squence_infos": evaluation_sequence_infos}

        return evaluation_info

    def _setup(self, configuration):
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

    def objective_function(self, configuration: Dict, cutoff_agent_per_sequence: Optional[int, float] = 600, **kwargs) \
            -> Dict:
        raise NotImplementedError()

    def objective_function_test(self, configuration: Dict, cutoff_agent_per_sequence: Optional[int, float] = 600,
                                **kwargs) -> Dict:
        return self.objective_function(configuration, cutoff_agent_per_sequence=cutoff_agent_per_sequence, **kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'Learna',
                'references': ['Frederic Runge and Danny Stoll and Stefan Falkner and Frank Hutter',
                               'Learning to Design {RNA} (ICLR) 2019',
                               'https://ml.informatik.uni-freiburg.de/papers/19-ICLR-Learning-Design-RNA.pdf'],
                'note': ''}

    @staticmethod
    def get_configuration_space(seed: int = 0) -> CS.ConfigurationSpace:
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
    def _fill_config(config: Dict) -> Dict:
        """ Helper function to fill the configuration space with the missing parameters """
        config["conv_size1"] = 1 + 2 * config["conv_radius1"]
        if config["conv_radius1"] == 0:
            config["conv_size1"] = 0
        del config["conv_radius1"]

        config["conv_size2"] = 1 + 2 * config["conv_radius2"]
        if config["conv_radius2"] == 0:
            config["conv_size2"] = 0
        del config["conv_radius2"]

        if config["conv_size1"] != 0:
            min_state_radius = config["conv_size1"] + config["conv_size1"] - 1
            max_state_radius = 32
            config["state_radius"] = int(min_state_radius
                                         + (max_state_radius - min_state_radius) * config["state_radius_relative"])
        else:
            min_state_radius = config["conv_size2"] + config["conv_size2"] - 1
            max_state_radius = 32
            config["state_radius"] = int(min_state_radius
                                         + (max_state_radius - min_state_radius) * config["state_radius_relative"])
        del config["state_radius_relative"]

        config["restart_timeout"] = None

        return config


class Learna(BaseLearna):

    def __init__(self, data_path: Union[str, Path]):
        super(Learna, self).__init__(data_path)

    def objective_function(self, configuration: Dict, cutoff_agent_per_sequence: Optional[int, float] = 600, **kwargs) \
            -> Dict:
        config, network_config, agent_config, env_config = self._setup(configuration)

        start_time = time()
        validation_info = self._validate(evaluation_timeout=cutoff_agent_per_sequence,
                                         restore_path=None,
                                         stop_learning=False,
                                         restart_timeout=config["restart_timeout"],
                                         network_config=network_config,
                                         agent_config=agent_config,
                                         env_config=env_config)
        cost = time() - start_time

        return {'function_value': validation_info["sum_of_min_distances"],
                'cost': cost,
                'num_solved': validation_info["num_solved"],
                'sum_of_first_distances': validation_info["sum_of_first_distances"],
                'squence_infos': validation_info["squence_infos"]}


class MetaLearna(BaseLearna):
    def __init__(self, data_path: Union[str, Path]):
        super(MetaLearna, self).__init__(data_path)
        self.config = hpolib.config.config_file

    def objective_function(self, configuration: Dict, cutoff_agent_per_sequence: Optional[int, float] = 600, **kwargs) \
            -> Dict:

        config_as_str = _replace_multiple_char(str(configuration),
                                               [', ', ': ', '\'', '\"', '\n'],
                                               ['-',  '-',  '',   '',   ''])
        tmp_dir = self.config.cache_dir / config_as_str

        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True, parents=True)

        config, network_config, agent_config, env_config = self._setup(configuration)
        start_time = time()
        cost = 10 ** 10
        try:
            train_info = self._train(budget=cutoff_agent_per_sequence,
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
        except Exception as e:
            raise e

        return {'function_value': validation_info["sum_of_min_distances"],
                'cost': cost,
                'train_info': train_info,
                'validation_info': validation_info}

    def _train(self, budget: Union[int, float], tmp_dir: Path, network_config: NetworkConfig,
               agent_config: AgentConfig, env_config: RnaDesignEnvironmentConfig) -> Dict:
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
        train_sequence_infos = {}
        train_sum_of_min_distances = 0
        train_sum_of_last_distances = 0
        train_num_solved = 0

        for r in train_results.values():
            sequence_id = r[0].target_id
            r.sort(key=lambda e: e.time)

            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, r)))

            train_sum_of_min_distances += dists.min()
            train_sum_of_last_distances += dists[-1]

            train_num_solved += dists.min() == 0.0

            train_sequence_infos[sequence_id] = {"num_episodes": len(r),
                                                 "min_distance": float(dists.min()),
                                                 "last_distance": float(dists[-1])}

        train_info = {"num_solved": int(train_num_solved),
                      "sum_of_min_distances": float(train_sum_of_min_distances),
                      "sum_of_last_distances": float(train_sum_of_last_distances),
                      "squence_infos": train_sequence_infos}

        return train_info

    @staticmethod
    def _process_train_results(train_results: List) -> Dict:
        results_by_sequence = {}
        for r in train_results:
            for s in r:
                if not s.target_id in results_by_sequence:
                    results_by_sequence[s.target_id] = [s]
                else:
                    results_by_sequence[s.target_id].append(s)

        return results_by_sequence
