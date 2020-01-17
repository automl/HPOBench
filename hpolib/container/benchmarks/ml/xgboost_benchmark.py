"""
@author: Stefan Staeglich
"""

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostOnMnist(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        # bName must be the exact same as the suffix in the recipe name (Singuarity.XGBoostOnMnist)
        self.bName = "XGBoostOnMnist"
        self._setup(**kwargs)

