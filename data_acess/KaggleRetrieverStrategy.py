from .IDataRetrieverStrategy import IDataRetrieverStrategy
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import pandas as pd


class KaggleRetrieverStrategy(IDataRetrieverStrategy):
    def get_data(self, **kwargs):
        try:

            dataset_name = kwargs['dataset_name']
            save_path = kwargs['save_path']

            # Autenticar
            api = KaggleApi()
            api.authenticate()
            
            # Baixar dataset
            print(f"Baixando dataset {dataset_name}...")
            api.dataset_download_files(dataset_name, 
                                        path=save_path, 
                                        unzip=True)
            print("Download completo!")
            
            # Listar arquivos baixados
            import os
            print("\nArquivos baixados:")
            for file in os.listdir(save_path):
                print(f"- {file}")
            
        except Exception as e:
            print(f"Erro ao baixar dataset: {e}")
