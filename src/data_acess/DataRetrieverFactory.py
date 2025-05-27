from .IDataRetrieverStrategy import IDataRetrieverStrategy

class DataRetrieverFactory:
    def __init__(self, strategy: IDataRetrieverStrategy):
        self._strategy = strategy

    def get_data(self, **kwargs):
        data = self._strategy.get_data(**kwargs)
        return(data)