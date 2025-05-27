from .ParamGridStrategy import ParamGridStrategy
from sklearn.neighbors import KNeighborsClassifier


class KNNStrategy(ParamGridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(
                             KNeighborsClassifier() 
                            ,n_neighbors=[3, 5, 7, 9]          # Número de vizinhos
                            ,weights=['uniform', 'distance']    # Peso das amostras
                            ,algorithm=['auto', 'ball_tree', 'kd_tree']  # Algoritmo de busca
                            ,p=[1, 2]                          # 1=Manhattan, 2=Euclidiana
                            ,leaf_size=[10, 30, 50]             # Para ball_tree/kd_tree
                           )
            .build()
        )
