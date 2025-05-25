import matplotlib.pyplot as plt
from .IPlotStrategy import IPlotStrategy

class PlotFactory:
    def __init__(self, strategy: IPlotStrategy):
        self._strategy = strategy

    def create_plot(self, data, **kwargs):
        self._strategy.plot(data, **kwargs)
        plt.tight_layout()
        plt.show()
