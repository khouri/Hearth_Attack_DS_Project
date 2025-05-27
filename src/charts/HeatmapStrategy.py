from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt

class HeatmapStrategy(IPlotStrategy):
    def plot(self, data, x, y, title=""):
        pivot_table = data.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
        plt.title(title)