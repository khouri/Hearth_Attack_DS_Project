class RandomForestStrategy(ParamGridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(
                            RandomForestClassifier()
                            n_estimators=[50, 100, 200],      # Número de árvores
                            max_depth=[5, 10, None],           # Profundidade máxima
                            min_samples_split=[2, 5, 10],      # Mínimo para dividir um nó
                            min_samples_leaf=[1, 2, 4],        # Mínimo em folhas
                            max_features=['sqrt', 'log2'],     # Features por split
                            bootstrap=[True, False],           # Amostras com reposição
                            class_weight=['balanced', None]    # Para dados desbalanceados
                           )
            .build()
        )