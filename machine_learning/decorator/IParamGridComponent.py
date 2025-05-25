from abc import ABC, abstractmethod

class IParamGridComponent(ABC):
    """Interface base para os componentes do param_grid"""
    def get_params(self) -> list:
        pass