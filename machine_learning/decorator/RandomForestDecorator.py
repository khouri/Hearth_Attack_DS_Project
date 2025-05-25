from .ParamGridDecorator import ParamGridDecorator
from sklearn.ensemble import RandomForestClassifier


class RandomForestDecorator(ParamGridDecorator):
    """Adiciona configurações de RandomForest ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10]
        })
        return params