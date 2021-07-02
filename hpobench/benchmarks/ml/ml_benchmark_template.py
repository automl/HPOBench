import time
import openml
import numpy as np
import pandas as pd
import ConfigSpace as CS
from typing import Union, Dict

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, f1_score, \
    top_k_accuracy_score, balanced_accuracy_score

from hpobench.abstract_benchmark import AbstractBenchmark


metrics = dict(
    #TODO: decide on metrics generalized for different datasets
    acc=accuracy_score,
    bal_acc=balanced_accuracy_score,
    f1=f1_score,
    # roc=roc_auc_score,
    # topk=top_k_accuracy_score
)
metrics_kwargs = dict(
    #TODO: decide on metric parameters
    acc=dict(),
    bal_acc=dict(),
    f1=dict(average="weighted"),
    # roc=dict(average="weighted"),
    # topk=dict()
)


class Benchmark(AbstractBenchmark):
    _issue_tasks = [3917, 3945]

    def __init__(
            self,
            task_id: Union[int, None] = None,
            seed: Union[int, None] = None,  # Union[np.random.RandomState, int, None] = None,
            valid_size: float = 0.33,
            fidelity_choice: int = 1
    ):
        self.seed = seed if seed is not None else np.random.randint(1, 10 ** 6)
        self.rng = check_random_state(self.seed)
        super(Benchmark, self).__init__(rng=seed)

        self.task_id = task_id
        self.valid_size = valid_size
        self.scorers = dict()
        for k, v in metrics.items():
            self.scorers[k] = make_scorer(v, **metrics_kwargs[k])
        # self.scorers = make_scorer(accuracy_score)

        # Data variables
        self.train_X = None
        self.valid_X = None
        self.test_X = None
        self.train_y = None
        self.valid_y = None
        self.test_y = None
        self.train_idx = None
        self.test_idx = None
        self.task = None
        self.dataset = None
        self.preprocessor = None
        self.lower_bound_train_size = None
        self.load_data_from_openml()

        # Observation and fidelity spaces
        self.fidelity_choice = fidelity_choice
        self.z_cs = self.get_fidelity_space(self.seed, self.fidelity_choice)
        self.x_cs = self.get_configuration_space(self.seed)

    @staticmethod
    def get_configuration_space(seed=None):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()

    @staticmethod
    def get_fidelity_space(seed=None, fidelity_choice=1):
        """Fidelity space available --- specifies the fidelity dimensions

        If fidelity_choice is 0
            Fidelity space is the maximal fidelity, akin to a black-box function
        If fidelity_choice is 1
            Fidelity space is a single fidelity, in this case the number of trees (n_estimators)
        If fidelity_choice is 2
            Fidelity space is a single fidelity, in this case the fraction of dataset (subsample)
        If fidelity_choice is >2
            Fidelity space is multi-multi fidelity, all possible fidelities
        """
        raise NotImplementedError()

    def get_config(self, size=None):
        """Samples configuration(s) from the (hyper) parameter space
        """
        if size is None:  # return only one config
            return self.x_cs.sample_configuration()
        return [self.x_cs.sample_configuration() for i in range(size)]

    def get_fidelity(self, size=None):
        """Samples candidate fidelities from the fidelity space
        """
        if size is None:  # return only one config
            return self.z_cs.sample_configuration()
        return [self.z_cs.sample_configuration() for i in range(size)]

    def _convert_labels(self, labels):
        """Converts boolean labels (if exists) to strings
        """
        label_types = list(map(lambda x: isinstance(x, bool), labels))
        if np.all(label_types):
            _labels = list(map(lambda x: str(x), labels))
            if isinstance(labels, pd.Series):
                labels = pd.Series(_labels, index=labels.index)
            elif isinstance(labels, np.array):
                labels = np.array(labels)
        return labels

    def load_data_from_openml(self, valid_size=None, verbose=False):
        """Fetches data from OpenML and initializes the train-validation-test data splits

        The validation set is fixed till this function is called again or explicitly altered
        """
        # fetches task
        self.task = openml.tasks.get_task(self.task_id, download_data=False)
        # fetches dataset
        self.dataset = openml.datasets.get_dataset(self.task.dataset_id, download_data=False)
        if verbose:
            print(self.task, '\n')
            print(self.dataset, '\n')

        # loads full data
        X, y, categorical_ind, feature_names = self.dataset.get_data(
            target=self.task.target_name, dataset_format="dataframe"
        )
        categorical_ind = np.array(categorical_ind)
        (cat_idx,) = np.where(categorical_ind)
        (cont_idx,) = np.where(~categorical_ind)

        # splitting dataset into train and test (10% test)
        # train-test split is fixed for a task and its associated dataset
        self.train_idx, self.test_idx = self.task.get_train_test_split_indices()
        train_x = X.iloc[self.train_idx]
        train_y = y.iloc[self.train_idx]
        self.test_X = X.iloc[self.test_idx]
        self.test_y = y.iloc[self.test_idx]

        # splitting training into training and validation
        # validation set is fixed till this function is called again or explicitly altered
        valid_size = self.valid_size if valid_size is None else valid_size
        self.train_X, self.valid_X, self.train_y, self.valid_y = train_test_split(
            train_x, train_y, test_size=valid_size,
            shuffle=True, stratify=train_y, random_state=self.rng
        )

        # preprocessor to handle missing values, categorical columns encodings,
        # and scaling numeric columns
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(sparse=False, handle_unknown="ignore")),
                    cat_idx.tolist(),
                ),
                (
                    "cont",
                    make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()),
                    cont_idx.tolist(),
                )
            ])
        )
        if verbose:
            print("Shape of data pre-preprocessing: {}".format(train_X.shape))

        # preprocessor fit only on the training set
        self.train_X = self.preprocessor.fit_transform(self.train_X)
        # applying preprocessor built on the training set, across validation and test splits
        self.valid_X = self.preprocessor.transform(self.valid_X)
        self.test_X = self.preprocessor.transform(self.test_X)
        # converting boolean labels to strings
        self.train_y = self._convert_labels(self.train_y)
        self.valid_y = self._convert_labels(self.valid_y)
        self.test_y = self._convert_labels(self.test_y)

        # Similar to (https://arxiv.org/pdf/1605.07079.pdf)
        # use 10 times the number of classes as lower bound for the dataset fraction
        n_classes = len(self.task.class_labels)
        self.lower_bound_train_size = (10 * n_classes) / self.train_X.shape[0]
        self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))

        if verbose:
            print("Shape of data post-preprocessing: {}".format(train_X.shape), "\n")

        if verbose:
            print("\nTraining data (X, y): ({}, {})".format(self.train_X.shape, self.train_y.shape))
            print("Validation data (X, y): ({}, {})".format(self.valid_X.shape, self.valid_y.shape))
            print("Test data (X, y): ({}, {})".format(self.test_X.shape, self.test_y.shape))
            print("\nData loading complete!\n")
        return

    def shuffle_data_idx(self, train_id=None, ng=None):
        rng = self.rng if rng is None else rng
        train_idx = self.train_idx if train_idx is None else train_idx
        rng.shuffle(train_idx)
        return train_idx

    def init_model(self, config, fidelity=None, rng=None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        raise NotImplementedError()

    def _raw_objective(self, config, fidelity, shuffle, rng, eval="valid"):
        # initializing model
        model = self.init_model(config, fidelity, rng)

        # preparing data
        if eval == "valid":
            train_X = self.train_X
            train_y = self.train_y
            train_idx = self.train_idx
        else:
            train_X = np.vstack((self.train_X, self.valid_X))
            train_y = pd.concat((self.train_y, self.valid_y))
            train_idx = np.arange(len(train_X))

        # shuffling data
        if shuffle:
            train_idx = self.shuffle_data_idx(train_idx, rng)
            train_X = train_X.iloc[train_idx]
            train_y = train_y.iloc[train_idx]

        # subsample here:
        # application of the other fidelity to the dataset that the model interfaces
        train_idx = self.rng.choice(
            np.arange(len(train_X)), size=int(
                fidelity['subsample'] * len(train_X)
            )
        )
        # fitting the model with subsampled data
        start = time.time()
        model.fit(train_X[train_idx], train_y.iloc[train_idx])
        model_fit_time = time.time() - start
        # computing statistics on training data
        scores = dict()
        score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            scores[k] = v(model, train_X, train_y)
            score_cost[k] = time.time() - _start
        train_loss = 1 - scores["acc"]
        return model, model_fit_time, train_loss, scores, score_cost

    def objective(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._raw_objective(
            configuration, fidelity, shuffle, rng
        )
        val_scores = dict()
        val_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            val_scores[k] = v(model, self.valid_X, self.valid_y)
            val_score_cost[k] = time.time() - _start
        val_loss = 1 - val_scores["acc"]

        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        val_loss = 1 - test_scores["acc"]

        info = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'val_scores': val_scores,
            'val_costs': val_score_cost,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity.get_dictionary(),
            'config': configuration.get_dictionary()
        }

        return {
            'function_value': info['val_loss'],
            'cost': model_fit_time + info['train_costs']['acc'] + info['val_costs']['acc'],
            'info': info
        }

    def objective_test(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """
        model, model_fit_time, train_loss, train_scores, train_score_cost = self._raw_objective(
            configuration, fidelity, shuffle, rng, eval="test"
        )
        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        test_loss = 1 - test_scores["acc"]

        info = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'model_cost': model_fit_time,
            'train_scores': train_scores,
            'train_costs': train_score_cost,
            'test_scores': test_scores,
            'test_costs': test_score_cost,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity.get_dictionary(),
            'config': configuration.get_dictionary()
        }

        return {
            'function_value': info['test_loss'],
            'cost': model_fit_time + info['train_costs']['acc'] + info['test_costs']['acc'],
            'info': info
        }

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        return dict()

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function_test(
            self,
            configuration: Union[CS.Configuration, Dict],
            fidelity: Union[CS.Configuration, Dict, None] = None,
            shuffle: bool = False,
            rng: Union[np.random.RandomState, int, None] = None,
            **kwargs
    ) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the test set
        """
        return dict()

    def get_meta_information(self):
        """ Returns the meta information for the benchmark """
        pass
