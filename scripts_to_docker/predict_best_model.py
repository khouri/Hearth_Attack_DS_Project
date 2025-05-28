from dotenv import load_dotenv, find_dotenv
import os
import sys 

_ = load_dotenv(find_dotenv())

project_path = os.environ["PROJECT_ABS_PATH"]
sys.path.append(os.path.abspath(os.path.join(project_path)))


from joblib import load
import pandas as pd
from pathlib import Path


def __load_model(path):
    return(load(path))
pass


def main(instance_tobe_scored):

    path = Path(os.environ["PROJECT_ABS_PATH"]) / "serialized_models" / "Random_forest.joblib"
    model = __load_model(path)
    score = model.predict_proba(instance_tobe_scored)[:, 1]

    return(score)
pass


#TODO, trocar por um serviço
if __name__ == "__main__":
    import gc
    gc.collect()
    
    project_path = Path(os.environ["PROJECT_ABS_PATH"])
    test_path = project_path + "data" / "test_predictors_FE.csv"

    test_predictors = pd.read_csv(test_path, 
                                  sep = ';', 
                                  index_col=0)
                                  
    first_instance = test_predictors.iloc[:1]
    print(type(first_instance))
    print(first_instance)

    score = main(first_instance)
    print("First instance score {0}".format(score))

pass

