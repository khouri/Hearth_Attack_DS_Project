from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class IPlotStrategy(ABC):
    @abstractmethod
    def plot(self, data, **kwargs):
        pass