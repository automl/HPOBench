"""
Changelog:
==========
0.0.4
* Set the lower bound of the hp `likelihood_ratio_clipping` to a small number instead of 0.
  The PPO agent does not accept a value of 0 here and will raise an error.
* Pass the hp `entropy_regularization` to the agent.
* Add the hp `entropy_regularization` to the ConfigSpace of the CartpoleFull Benchmark.

0.0.3
* New container release due to a general change in the communication between container and HPOBench.
  Works with HPOBench >= v0.0.8

0.0.2:
* Standardize the structure of the meta information
* Suppress unnecessary tensorforce logging messages

0.0.1:
* First implementation
"""

import logging
import time
from typing import Union, Dict

import ConfigSpace as CS
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402
from tensorforce.agents import PPOAgent  # noqa: E402
from tensorforce.contrib.openai_gym import OpenAIGym  # noqa: E402
from tensorforce.execution import Runner  # noqa: E402

from hpobench.abstract_benchmark import AbstractBenchmark  # noqa: E402
from hpobench.util import rng_helper  # noqa: E402

__version__ = '0.0.4'

logger = logging.getLogger('CartpoleBenchmark')
tf.logging.set_verbosity(tf.logging.ERROR)


class CartpoleBase(AbstractBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, defaults: Union[Dict, None] = None,
                 max_episodes: Union[int, None] = 3000):
        """
        Base benchmark for "cartpole" benchmark. In this benchmark a PPO agent tries to solve the cartpole task.

        Parameters
        ----------
        rng : int,None,np.RandomState
            RandomState for the experiment
        defaults : dict, None
            default configuration used for the PPO agent
        max_episodes : int, None
            limit of the length of a episode for the cartpole runner. Defaults to 3000
        """

        logger.warning('This Benchmark is not deterministic.')
        super(CartpoleBase, self).__init__()

        self.rng = rng_helper.get_rng(rng=rng)
        tf.random.set_random_seed(0)
        np.random.seed(0)
        self.env = OpenAIGym('CartPole-v0', visualize=False)
        self.avg_n_episodes = 20
        self.max_episodes = max_episodes

        self.defaults = {"n_units_1": 64,
                         "n_units_2": 64,
                         "batch_size": 64,
                         "learning_rate": 1e-3,
                         "discount": 0.99,
                         "likelihood_ratio_clipping": 0.2,
                         "activation_1": "tanh",
                         "activation_2": "tanh",
                         "optimizer_type": "adam",
                         "optimization_steps": 10,

                         "baseline_mode": "states",
                         "baseline_n_units_1": 64,
                         "baseline_n_units_2": 64,
                         "baseline_learning_rate": 1e-3,
                         "baseline_optimization_steps": 10,
                         "baseline_optimizer_type": "adam"}

        if defaults is not None:
            self.defaults.update(defaults)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Returns the CS.ConfigurationSpace of the benchmark. """
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Creates a ConfigSpace.ConfigurationSpace containing all fidelity parameters for
        all Cartpole Benchmarks

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
            CS.UniformIntegerHyperparameter('budget', lower=1, upper=9, default_value=9)
        ])

        return fidel_space

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Trains a Tensorforce RL agent on the cartpole experiment. This benchmark was used in the experiments for the
        BOHB-paper (see references). A more detailed explanations can be found there.

        The budget describes how often the agent is trained on the experiment.
        It returns the average number of the length of episodes.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : average episode length
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        tf.random.set_random_seed(self.rng.randint(1, 100000))
        np.random.seed(self.rng.randint(1, 100000))

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        new_config = self.defaults
        new_config.update(configuration)
        configuration = new_config

        start_time = time.time()

        network_spec = [{'type': 'dense',
                         'size': configuration["n_units_1"],
                         'activation': configuration['activation_1']},
                        {'type': 'dense',
                         'size': configuration["n_units_2"],
                         'activation': configuration['activation_2']}]

        converged_episodes = []

        for _ in range(fidelity["budget"]):
            agent = PPOAgent(states=self.env.states,
                             actions=self.env.actions,
                             network=network_spec,
                             update_mode={'unit': 'episodes', 'batch_size': configuration["batch_size"]},
                             step_optimizer={'type': configuration["optimizer_type"],
                                             'learning_rate': configuration["learning_rate"]},
                             optimization_steps=configuration["optimization_steps"],
                             discount=configuration["discount"],
                             baseline_mode=configuration["baseline_mode"],
                             baseline={"type": "mlp",
                                       "sizes": [configuration["baseline_n_units_1"],
                                                 configuration["baseline_n_units_2"]]},
                             baseline_optimizer={"type": "multi_step",
                                                 "optimizer": {"type": configuration["baseline_optimizer_type"],
                                                               "learning_rate":
                                                                   configuration["baseline_learning_rate"]},
                                                 "num_steps": configuration["baseline_optimization_steps"]},
                             likelihood_ratio_clipping=configuration["likelihood_ratio_clipping"],
                             entropy_regularization=configuration["entropy_regularization"],
                             )

            def episode_finished(record):
                # Check if we have converged
                return np.mean(record.episode_rewards[-self.avg_n_episodes:]) != 200

            runner = Runner(agent=agent, environment=self.env)
            runner.run(episodes=self.max_episodes, max_episode_timesteps=200, episode_finished=episode_finished)
            converged_episodes.append(len(runner.episode_rewards))

        cost = time.time() - start_time

        return {'function_value': np.mean(converged_episodes),
                'cost': cost,
                'info': {'max_episodes': self.max_episodes,
                         'all_runs': converged_episodes,
                         'fidelity': fidelity
                         }
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the cartpole benchmark. Use the full budget.
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : average episode length
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """

        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'Cartpole',
                'references': ['@InProceedings{falkner-icml-18,'
                               'title     = {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},'
                               'url       = http://proceedings.mlr.press/v80/falkner18a.html'
                               'author    = {Falkner, Stefan and Klein, Aaron and Hutter, Frank}, '
                               'booktitle = {Proceedings of the 35th International Conference on Machine Learning},'
                               'pages     = {1436 - -1445},'
                               'year      = {2018}}'],
                'code': 'https://github.com/automl/HPOlib1.5/blob/development/hpolib/benchmarks/rl/cartpole.py',
                'note': 'This benchmark is not deterministic, since the gym environment is not deterministic.'
                        ' Also, often the benchmark is already converged after 1000 episodes.'
                        ' Increasing the budget \"max_episodes\" may lead to the same results.'}


class CartpoleFull(CartpoleBase):
    """Cartpole experiment on full configuration space"""

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space for this benchmark
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
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter("n_units_1", lower=8, default_value=64, upper=64, log=True),
            CS.UniformIntegerHyperparameter("n_units_2", lower=8, default_value=64, upper=64, log=True),
            CS.UniformIntegerHyperparameter("batch_size", lower=8, default_value=64, upper=256, log=True),
            CS.UniformFloatHyperparameter("learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True),
            CS.UniformFloatHyperparameter("discount", lower=0, default_value=.99, upper=1),
            CS.UniformFloatHyperparameter("likelihood_ratio_clipping", lower=1e-7, default_value=.2, upper=1),
            CS.UniformFloatHyperparameter("entropy_regularization", lower=0, default_value=0.01, upper=1),
            CS.CategoricalHyperparameter("activation_1", ["tanh", "relu"]),
            CS.CategoricalHyperparameter("activation_2", ["tanh", "relu"]),
            CS.CategoricalHyperparameter("optimizer_type", ["adam", "rmsprop"]),
            CS.UniformIntegerHyperparameter("optimization_steps", lower=1, default_value=10, upper=10),
            CS.CategoricalHyperparameter("baseline_mode", ["states", "network"]),
            CS.UniformIntegerHyperparameter("baseline_n_units_1", lower=8, default_value=64, upper=128, log=True),
            CS.UniformIntegerHyperparameter("baseline_n_units_2", lower=8, default_value=64, upper=128, log=True),
            CS.UniformFloatHyperparameter("baseline_learning_rate",
                                          lower=1e-7, default_value=1e-3, upper=1e-1, log=True),
            CS.UniformIntegerHyperparameter("baseline_optimization_steps", lower=1, default_value=10, upper=10),
            CS.CategoricalHyperparameter("baseline_optimizer_type", ["adam", "rmsprop"]),
        ])
        return cs

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = CartpoleBase.get_meta_information()
        meta_information['description'] = 'Cartpole with full configuration space'
        return meta_information


class CartpoleReduced(CartpoleBase):
    """Cartpole experiment on smaller configuration space"""

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space for this benchmark
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
        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter("n_units_1", lower=8, default_value=64, upper=128, log=True),
            CS.UniformIntegerHyperparameter("n_units_2", lower=8, default_value=64, upper=128, log=True),
            CS.UniformIntegerHyperparameter("batch_size", lower=8, default_value=64, upper=256, log=True),
            CS.UniformFloatHyperparameter("learning_rate", lower=1e-7, default_value=1e-3, upper=1e-1, log=True),
            CS.UniformFloatHyperparameter("discount", lower=0, default_value=.99, upper=1),
            CS.UniformFloatHyperparameter("likelihood_ratio_clipping", lower=1e-7, default_value=.2, upper=1),
            CS.UniformFloatHyperparameter("entropy_regularization", lower=0, default_value=0.01, upper=1),
        ])
        return cs

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = CartpoleBase.get_meta_information()
        meta_information['description'] = 'Cartpole with reduced configuration space'
        return meta_information
