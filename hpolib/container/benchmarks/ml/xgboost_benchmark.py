"""
@author: Stefan Staeglich
"""

from hpolib.container.client_abstract_benchmark import AbstractBenchmarkClient


class XGBoostOnMnist(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "XGBoostOnMnist"
        self._setup(**kwargs)

