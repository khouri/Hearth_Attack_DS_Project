from .IParamGridComponent import IParamGridComponent

class ParamGridDecorator(IParamGridComponent):
    """Classe base para todos os decoradores"""
    _component: IParamGridComponent = None

    def __init__(self, component: IParamGridComponent) -> None:
        self._component = component

    @property
    def component(self) -> IParamGridComponent:
        return self._component

    def get_params(self) -> list:
        return self._component.get_params()
