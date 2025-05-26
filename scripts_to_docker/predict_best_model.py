from joblib import load
import pandas as pd


def __load_model(path):
    return(load(path))
pass


def main(instance_tobe_scored):
    print('main')
    model = __load_model('../model_serializable/Random_forest.joblib')
    score  = model.predict_proba(instance_tobe_scored)[:, 1]

    # predict
    return(score)
pass


#TODO, trocar por um serviço
if __name__ == "__main__":

    test_predictors = pd.read_csv('../data/test_predictors_FE.csv', 
                                  sep = ';', 
                                  index_col=0)
                                  
    first_instance = test_predictors.iloc[:1]
    print(type(first_instance))
    print(first_instance)

    score = main(first_instance)
    print("First instance score {0}".format(score))

pass

