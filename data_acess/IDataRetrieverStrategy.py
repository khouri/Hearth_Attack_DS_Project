from abc import ABC, abstractmethod

class IDataRetrieverStrategy(ABC):
    @abstractmethod
    def get_data(self, **kwargs):
        pass