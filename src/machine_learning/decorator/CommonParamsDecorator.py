from .ParamGridDecorator import ParamGridDecorator
from .IParamGridComponent import IParamGridComponent


class CommonParamsDecorator(ParamGridDecorator):
    """Adiciona parâmetros comuns a todos os classificadores"""
    def __init__(self, component: IParamGridComponent, random_states=None):
        super().__init__(component)
        self.random_states = random_states or [42]

    def get_params(self) -> list:
        params = self.component.get_params()
        for config in params:
            config['classifier__random_state'] = self.random_states
        return params