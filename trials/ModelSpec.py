import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Union

import numpy as np
from nasbench.api import ModelSpec

from trials.Constants import nasbench


@dataclass
class ModelData:
    """
    Wraps up resulting data from NASBench.query in a class for code cleanliness.
    """

    hash: str
    matrix: List[List[int]]
    operations: List[str]
    parameters: int
    train_time: float
    train_accuracy: float
    valid_accuracy: float
    test_accuracy: float

    @property
    def total_accuracy(self) -> float:
        return (self.test_accuracy + self.valid_accuracy) / 2


class SpecWrapper(ModelSpec):

    id: int
    id_iter = itertools.count()

    """
    Wraps a model to allow easier access to the resulting data of the model.
    """

    def __init__(
        self,
        matrix: Union[str, np.ndarray],
        ops: List[str] = None,
        stop_halfway: bool = False,
    ):
        self.id = next(SpecWrapper.id_iter)
        self.stop_halfway = stop_halfway

        if isinstance(matrix, str):
            spec = nasbench.get_metrics_from_hash(matrix)
            matrix: np.ndarray = spec[0]["module_adjacency"]
            ops = spec[0]["module_operations"]
            super().__init__(matrix, ops)
        else:
            assert ops is not None
            super().__init__(matrix, ops)

    # defining how to compare two specs is all we need to sort
    # we'll define basede on "total accuracy", which we assume
    # to be test accuracy + validation accuracy. basically we're
    # dual optimizing with this
    def __lt__(self, other: "SpecWrapper"):
        if not (isinstance(other, SpecWrapper)):
            return False

        s_data = self.get_data()
        o_data = other.get_data()

        s_total = s_data.test_accuracy + s_data.valid_accuracy
        o_total = o_data.test_accuracy + o_data.valid_accuracy
        return s_total < o_total

    def get_hash(self):
        return self.hash_spec(nasbench.config["available_ops"])

    @lru_cache(maxsize=1)
    def get_data(self):
        """
        Get resultant data from NASBench.
        The LRU cache ensures we don"t add time to the budget counters multiple times.
        :return:
        """
        data = nasbench.query(self, stop_halfway=self.stop_halfway)
        return ModelData(
            self.hash_spec(nasbench.config["available_ops"]),
            data["module_adjacency"],
            data["module_operations"],
            data["trainable_parameters"],
            data["training_time"],
            data["train_accuracy"],
            data["validation_accuracy"],
            data["test_accuracy"],
        )

    def __repr__(self):
        return self.get_hash()
