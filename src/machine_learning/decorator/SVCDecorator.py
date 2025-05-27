from .ParamGridDecorator import ParamGridDecorator
from sklearn.svm import SVC


class SVCDecorator(ParamGridDecorator):
    """Adiciona configurações de SVC ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [SVC()],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__C': [0.1, 1, 10]
        })
        return params