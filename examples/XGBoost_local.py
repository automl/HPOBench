from time import time
from hpolib.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark as Benchmark
from hpolib.util.openml_data_manager import get_openmlcc18_taskids

task_ids = get_openmlcc18_taskids()
for task_no, task_id in enumerate(task_ids):

    print(f'###################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ######################')
    if task_id == 167204:
        continue  # due to memory limits

    b = Benchmark(task_id=task_id)
    cs = b.get_configuration_space()
    start = time()
    num_configs = 1
    for i in range(num_configs):
        configuration = cs.sample_configuration()
        print(configuration)
        for n_estimator in [8, 64]:
            for subsample in [0.4, 1]:
                result_dict = b.objective_function(configuration, n_estimators=n_estimator, subsample=subsample)
                valid_loss = result_dict['function_value']
                train_loss = result_dict['train_loss']

                result_dict = b.objective_function_test(configuration, n_estimators=n_estimator)
                test_loss = result_dict['function_value']

                print(f'[{i+1}|{num_configs}] No Estimator: {n_estimator:3d} - Subsample Rate: {subsample:.1f} - Test {test_loss:.4f} '
                      f'- Valid {valid_loss:.4f} - Train {train_loss:.4f}')
    print(f'Done, took totally {time()-start:.2f}')
