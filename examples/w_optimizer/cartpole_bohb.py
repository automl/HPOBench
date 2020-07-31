"""
SMAC on Cartpole with BOHB
==========================

This example shows the usage of an Hyperparameter Tuner, such as BOHB on the cartpole benchmark.
BOHB is a combination of Bayesian optimization and Hyperband.

Please install the necessary dependencies via ``pip install .[singularity]`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps

"""
import logging
import pickle
from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

from hpolib.container.benchmarks.rl.cartpole import CartpoleReduced as Benchmark
from hpolib.util.example_utils import get_travis_settings, set_env_variables_to_use_only_one_core
from hpolib.util.rng_helper import get_rng

logger = logging.getLogger('BOHB on cartpole')
set_env_variables_to_use_only_one_core()


class CustomWorker(Worker):
    def __init__(self, seed, max_budget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.max_budget = max_budget

    def compute(self, config, budget, **kwargs):
        b = Benchmark(rng=self.seed)
        result_dict = b.objective_function(config, budget=int(budget))
        return {'loss': result_dict['function_value'],
                'info': {'cost': result_dict['cost'],
                         'budget': result_dict['budget']}}


def run_experiment(out_path, on_travis):

    settings = {'min_budget': 1,
                'max_budget': 5,  # Number of Agents, which are trained to solve the cartpole experiment
                'num_iterations': 10,  # Number of HB brackets
                'eta': 3,
                'output_dir': Path(out_path)
                }
    if on_travis:
        settings.update(get_travis_settings('bohb'))

    b = Benchmark(container_source='library://phmueller/automl',
                  container_name='cartpole')

    b.get_configuration_space(seed=1)
    settings.get('output_dir').mkdir(exist_ok=True)

    cs = b.get_configuration_space()
    seed = get_rng(rng=0)
    run_id = 'BOHB_on_cartpole'

    result_logger = hpres.json_result_logger(directory=str(settings.get('output_dir')), overwrite=True)

    ns = hpns.NameServer(run_id=run_id, host='localhost', working_directory=str(settings.get('output_dir')))
    ns_host, ns_port = ns.start()

    worker = CustomWorker(seed=seed,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          run_id=run_id,
                          max_budget=settings.get('max_budget'))
    worker.run(background=True)

    master = BOHB(configspace=cs,
                  run_id=run_id,
                  host=ns_host,
                  nameserver=ns_host,
                  nameserver_port=ns_port,
                  eta=settings.get('eta'),
                  min_budget=settings.get('min_budget'),
                  max_budget=settings.get('max_budget'),
                  result_logger=result_logger)

    result = master.run(n_iterations=settings.get('num_iterations'))
    master.shutdown(shutdown_workers=True)
    ns.shutdown()

    with open(settings.get('output_dir') / 'results.pkl', 'wb') as f:
        pickle.dump(result, f)

    id2config = result.get_id2config_mapping()
    incumbent = result.get_incumbent_id()
    inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
    inc_cfg = id2config[incumbent]['config']

    logger.info(f'Inc Config:\n{inc_cfg}\n'
                f'with Performance: {inc_value:.2f}')

    if not on_travis:
        benchmark = Benchmark(container_source='library://phmueller/automl')
        incumbent_result = benchmark.objective_function_test(configuration=inc_cfg,
                                                             budget=settings['max_budget'])
        print(incumbent_result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='HPOlib - BOHB',
                                     description='HPOlib3 with BOHB on Cartpole',
                                     usage='%(prog)s --out_path <string>')
    parser.add_argument('--out_path', default='./cartpole_smac_hb', type=str)
    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')
    args = parser.parse_args()

    run_experiment(out_path=args.out_path, on_travis=args.on_travis)
