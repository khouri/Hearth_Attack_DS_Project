import pandas as pd

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configura o Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from machine_learning.decorator.BaseParamGrid import BaseParamGrid
from machine_learning.decorator.CommonParamsDecorator import CommonParamsDecorator
from machine_learning.decorator.RandomForestDecorator import RandomForestDecorator

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from joblib import dump



def __kfold_strategy():
    skf = StratifiedKFold(
                        n_splits=5, 
                        shuffle=True, 
                        random_state=42
                    )
    return(skf)
pass

def __create_the_best_param_grid():

    decorated_grid = CommonParamsDecorator(
                                    RandomForestDecorator(
                                                            BaseParamGrid()
                                                        )
                                    ,random_states=[42]
                                )
    
    return(decorated_grid.get_params())
pass

def __grid_serch_strategy():

    # Definir as métricas que queremos
    scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='weighted'),
                'f1': make_scorer(f1_score, average='weighted')
            }

    pipe = Pipeline([
                    # Modelo stub (será substituído)
                    ('classifier', DummyClassifier())  
                    ])

    skf = __kfold_strategy()
    param_grid = __create_the_best_param_grid()

    grid = GridSearchCV(estimator = pipe,
                        param_grid = param_grid, 
                        cv=skf, 
                        refit='f1',
                        scoring=scoring)
    
    return(grid)
pass


def __load_train_data():

    project_path = os.environ["PROJECT_ABS_PATH"]

    train_path = project_path + "\\data\\train_predictors_FE.csv"
    label_path = project_path + "\\data\\train_label_FE.csv"

    train_predictors = pd.read_csv(train_path, sep = ';', index_col=0)
    train_label = pd.read_csv(label_path, sep = ';', index_col=0)

    return(train_predictors, train_label)
pass


def train_model():

    train_predictors, train_label = __load_train_data()
    train_label = train_label.squeeze()

    grid = __grid_serch_strategy()

    grid.fit(train_predictors, train_label)
    
    best_model = grid.best_estimator_
    path = os.environ["PROJECT_ABS_PATH"] + "\\model_serializable\\Random_forest.joblib"
    dump(best_model, path)

pass


def main():
    train_model()
pass


if __name__ == "__main__":
    import gc
    gc.collect()
    main()

    
    # project_path = os.environ["PROJECT_ABS_PATH"]
    # print("\n")
    # print("\n")
    # print("-------------")
    # print(project_path)
    # pat = "\\data\\train_predictors_FE.csv"
    # tmp = project_path + pat  
    # print(type(tmp))
    # print(tmp)
    # print("-------------")
    # print("\n")
    # print("\n")

pass