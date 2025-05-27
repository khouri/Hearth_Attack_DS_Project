from .ParamGridDecorator import ParamGridDecorator
from sklearn.linear_model import LogisticRegression


class LogisticRegressionDecorator(ParamGridDecorator):
    """Adiciona configurações de LogisticRegression ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear']
        })
        return params
