from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt

class BarPlotStrategy(IPlotStrategy):
    def plot(self, data, x, y, title="", xlabel="", ylabel=""):
        plt.bar(data[x], data[y], color='#4ECDC4')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--')