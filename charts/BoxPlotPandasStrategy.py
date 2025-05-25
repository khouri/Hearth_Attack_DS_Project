from .IPlotStrategy import IPlotStrategy
import matplotlib.pyplot as plt
import pandas as pd

class BoxPlotPandasStrategy(IPlotStrategy):
    def plot(self, data, x, y, title="", xlabel="", ylabel=""):
        # fig, ax = plt.subplots()
        
        data.boxplot(by = x 
                    ,column = y 
                    # ,xlabel = xlabel
                    # ,ylabel = ylabel
                    # ,title = title
                    ,grid = False)
        plt.title(title)  # Adicione depois
        plt.suptitle('')  # Remove o título automático do pandas
        # Adicionando Título ao gráfico
        # plt.title(title, loc="center", fontsize=18)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)

        # plt.show()


# df.boxplot(by ='day', column =['total_bill'], grid = False)
# # Tamanho do gráfico em polegadas
# plt.figure(figsize =(11, 6))

# #Plotando o boxplot das espécies em relação ao tamanho das sépalas
# bplots = plt.boxplot(dados_sepal_length,  vert = 0, patch_artist = False)

# # Adicionando Título ao gráfico
# plt.title("Boxplot da base de dados Íris", loc="center", fontsize=18)
# plt.xlabel("Comprimento das sépalas")
# plt.ylabel("Tipos de Flores")

# plt.show()