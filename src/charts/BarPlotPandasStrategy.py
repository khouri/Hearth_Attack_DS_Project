from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt
import pandas as pd

class BarPlotPandasStrategy(IPlotStrategy):
    def plot(self, dataframe, x, y, title="", xlabel="", ylabel=""):
        dataframe.plot(x 
                        ,kind = 'bar'
                        ,stacked = False
                        ,xlabel = xlabel
                        ,ylabel = ylabel
                        ,title = title
                        ,rot = 0
                        )
