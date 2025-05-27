from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt
import pandas as pd

class BoxPlotMatplotlibStrategy(IPlotStrategy):
    def plot(self, variavel, x, y, title="", xlabel="", ylabel=""):
        plt.boxplot(dataframe
                    ,vert = 1
                    )

        # Adicionando Título ao gráfico
        plt.title(title, loc="center", fontsize=18)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.show()



# # Tamanho do gráfico em polegadas
# plt.figure(figsize =(11, 6))

# #Plotando o boxplot das espécies em relação ao tamanho das sépalas
# bplots = plt.boxplot(dados_sepal_length,  vert = 0, patch_artist = False)

# # Adicionando Título ao gráfico
# plt.title("Boxplot da base de dados Íris", loc="center", fontsize=18)
# plt.xlabel("Comprimento das sépalas")
# plt.ylabel("Tipos de Flores")

# plt.show()