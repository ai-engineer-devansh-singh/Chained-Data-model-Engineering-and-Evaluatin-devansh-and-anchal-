from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        ...


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """

        """
        ...

    @abstractmethod
    def print_results(self) -> None:
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return
