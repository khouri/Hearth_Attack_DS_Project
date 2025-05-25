class ParamGridBuilder():

    def __init__(self):
        self.param_grid = []
    pass
    
    def add_classifier(self, classifier, **params):
        """
        Adiciona um classificador e seus hiperparâmetros ao param_grid.
        
        Args:
            classifier: Instância do classificador (ex: LogisticRegression()).
            params: Dicionário de hiperparâmetros (ex: C=[0.1, 1, 10]).
                    Prefixo 'pipeline__classifier__' é adicionado automaticamente.
        """
        classifier_entry = {
            'pipeline__classifier': [classifier],
            **{f'pipeline__classifier__{key}': value for key, value in params.items()}
        }
        self.param_grid.append(classifier_entry)
        
        return self  # Permite method chaining
    pass
    

    def build(self):
        """Retorna o param_grid construído."""
        return self.param_grid
    pass

pass