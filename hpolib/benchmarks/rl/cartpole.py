import logging
import time
from typing import Union, Optional, Dict

import ConfigSpace as CS
import numpy as np
import tensorflow as tf
from tensorforce.agents import PPOAgent
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.execution import Runner

from hpolib.abstract_benchmark import AbstractBenchmark
from hpolib.util import rng_helper

__version__ = '0.0.1'

logger = logging.getLogger('CartpoleBenchmark')


class CartpoleBase(AbstractBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None, defaults: Optional[Dict] = None,
                 max_episodes: Optional[int] = 3000):
        """
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

    @AbstractBenchmark._configuration_as_dict
    @AbstractBenchmark._check_configuration
    def objective_function(self, configuration: Union[Dict, CS.Configuration], budget: Optional[int] = 9,
                           rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a Tensorforce RL agent on the cartpole experiment. This benchmark was used in the experiments for the
        BOHB-paper (see references). A more detailed explanations can be found there.

        The budget describes how often the agent is trained on the experiment.
        It returns the average number of the length of episodes.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        budget : int, None
            Num Agents. Defaults to 9
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
            max_episodes : the maximum length of an episode
            budget : number of agents used
            all_runs : the episode length of all runs of all agents
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        tf.random.set_random_seed(self.rng.randint(1, 100000))
        np.random.seed(self.rng.randint(1, 100000))

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        c = self.defaults
        c.update(configuration)
        configuration = c

        start_time = time.time()

        network_spec = [{'type': 'dense',
                         'size': configuration["n_units_1"],
                         'activation': configuration['activation_1']},
                        {'type': 'dense',
                         'size': configuration["n_units_2"],
                         'activation': configuration['activation_2']}]

        converged_episodes = []

        for i in range(budget):
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
                             likelihood_ratio_clipping=configuration["likelihood_ratio_clipping"]
                             )

            def episode_finished(r):
                # Check if we have converged
                return np.mean(r.episode_rewards[-self.avg_n_episodes:]) != 200

            runner = Runner(agent=agent, environment=self.env)
            runner.run(episodes=self.max_episodes, max_episode_timesteps=200, episode_finished=episode_finished)
            converged_episodes.append(len(runner.episode_rewards))

        cost = time.time() - start_time

        return {'function_value': np.mean(converged_episodes),
                'cost': cost,
                'max_episodes': self.max_episodes,
                'budget': budget,
                'all_runs': converged_episodes}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config: Union[Dict, CS.Configuration], budget: Optional[int] = 9,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Validate a configuration on the cartpole benchmark. Use the full budget.
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        budget : int, None
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
            max_episodes : the maximum length of an episode
            budget : number of agents used
            all_runs : the episode length of all runs of all agents
        """
        return self.objective_function(config, budget=budget, rng=rng, **kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'Cartpole',
                'references': ['http://proceedings.mlr.press/v80/falkner18a.html'],
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
            CS.UniformFloatHyperparameter("likelihood_ratio_clipping", lower=0, default_value=.2, upper=1),
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
            CS.UniformFloatHyperparameter("likelihood_ratio_clipping", lower=0, default_value=.2, upper=1),
            CS.UniformFloatHyperparameter("entropy_regularization", lower=0, default_value=0.01, upper=1)
        ])
        return cs

    @staticmethod
    def get_meta_information() -> Dict:
        """ Returns the meta information for the benchmark """
        meta_information = CartpoleBase.get_meta_information()
        meta_information['description'] = 'Cartpole with reduced configuration space'
        return meta_information
