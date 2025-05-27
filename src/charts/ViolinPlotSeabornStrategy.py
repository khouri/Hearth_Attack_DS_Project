from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ViolinPlotSeabornStrategy(IPlotStrategy):
    def plot(self, data, x, y, hue, palette, title="", xlabel="", ylabel=""):
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=data, 
                        x=x,                    #cat
                        y=y,                    #num
                        hue=hue,                
                        palette=palette)        #cat
        plt.title(title)
        plt.show()