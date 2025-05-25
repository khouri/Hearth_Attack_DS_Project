from .IParamGridComponent import IParamGridComponent


class BaseParamGrid(IParamGridComponent):
    """Implementação concreta básica do param_grid"""
    def get_params(self) -> list:
        return []
