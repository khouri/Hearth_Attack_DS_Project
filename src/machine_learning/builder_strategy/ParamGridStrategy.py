from abc import ABC, abstractmethod
from .ParamGridBuilder import ParamGridBuilder


class ParamGridStrategy(ABC):
    @abstractmethod
    def build_grid(self, builder: 'ParamGridBuilder') -> list:
        pass