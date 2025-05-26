

The dataset used in this analysis was sourced from Kaggle:

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction



# how to build the image:

docker buildx build -t Hearth_attack_model --file Dockerfile_Hearth_attack_model .




# how to use poetry
poetry commands:

poetry env list
poetry env use C:\Users\PC\.conda\envs\gen_ai_31104\python.exe
poetry env remove --all
poetry add langchain-groq@0.3.0
poetry shell
poetry install 

# nao instala o projeto, apenas as dependencias
poetry install --no-root


poetry add scikit-learn

poetry add pandas 
poetry add numpy



poetry add seaborn
poetry add matplotlib
poetry add dotenv
poetry add zipfile
poetry add kaggle
poetry add seaborn
poetry add joblib


import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Lista de classificadores
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# Criar datasets
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    ("Moons", make_moons(noise=0.3, random_state=0)),
    ("Circles", make_circles(noise=0.2, factor=0.5, random_state=1)),
    ("Linear", linearly_separable),
]

# DataFrame para armazenar resultados
results = []

# Iterar sobre os datasets
for ds_name, ds in datasets:
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Iterar sobre os classificadores
    for name, clf in zip(names, classifiers):
        # Criar pipeline com StandardScaler
        pipeline = make_pipeline(StandardScaler(), clf)
        
        # Treinar e avaliar
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        # Adicionar resultados
        results.append({
            "Dataset": ds_name,
            "Classifier": name,
            "Accuracy": score
        })

# Criar DataFrame com os resultados
results_df = pd.DataFrame(results)

# Pivot table para melhor visualização
pivot_results = results_df.pivot(index="Classifier", columns="Dataset", values="Accuracy")
print(pivot_results)

# Retornar o DataFrame completo
results_df




--Definindo modelos no grid de parametros

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Pipeline com pré-processamento + placeholder para o classificador
pipeline = make_pipeline(
    StandardScaler(),
    None  # Será substituído pelo GridSearchCV
)

# Define os modelos e seus parâmetros
param_grid = [
    {
        'pipeline__classifier': [LogisticRegression()],
        'pipeline__classifier__C': [0.1, 1, 10]
    },
    {
        'pipeline__classifier': [RandomForestClassifier()],
        'pipeline__classifier__n_estimators': [50, 100]
    },
    {
        'pipeline__classifier': [SVC()],
        'pipeline__classifier__kernel': ['linear', 'rbf']
    }
]

# Adiciona o nome do passo do classificador ao pipeline
from sklearn.base import clone
pipeline.steps.append(('classifier', None))

# GridSearchCV testará todos os modelos
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Melhor modelo:", grid.best_estimator_)
1




class GridSearchBuilder:
    def __init__(self):
        self.param_grid = []

    def add_classifier(self, classifier, params: dict):
        self.param_grid.append({
            'pipeline__classifier': [classifier],
            **{f'pipeline__classifier__{k}': v for k, v in params.items()}
        })
        return self

    def build(self):
        return self.param_grid

# Uso:
builder = GridSearchBuilder()
param_grid = (
    builder
    .add_classifier(LogisticRegression(), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']})
    .add_classifier(RandomForestClassifier(), {'n_estimators': [50, 100]})
    .build()
)




from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class ParamGridBuilder:
    def __init__(self):
        self.param_grid = []
    
    def add_classifier(self, classifier, **params):
        """
        Adiciona um classificador e seus hiperparâmetros ao param_grid.
        
        Args:
            classifier: Instância do classificador (ex: LogisticRegression()).
            params: Dicionário de hiperparâmetros (ex: C=[0.1, 1, 10]).
                    Prefixo 'pipeline__classifier__' é adicionado automaticamente.
        """
        classifier_entry = {
            'pipeline__classifier': [classifier],
            **{f'pipeline__classifier__{key}': value for key, value in params.items()}
        }
        self.param_grid.append(classifier_entry)
        return self  # Permite method chaining
    
    def build(self):
        """Retorna o param_grid construído."""
        return self.param_grid

# --- Exemplo de Uso --- #
builder = ParamGridBuilder()

# Adiciona múltiplos classificadores com seus hiperparâmetros
param_grid = (
    builder
    .add_classifier(
        LogisticRegression(),
        C=[0.1, 1, 10],
        penalty=['l1', 'l2'],
        solver=['saga']
    )
    .add_classifier(
        RandomForestClassifier(),
        n_estimators=[50, 100],
        max_depth=[None, 5, 10]
    )
    .add_classifier(
        SVC(),
        C=[0.1, 1, 10],
        kernel=['linear', 'rbf'],
        gamma=['scale', 'auto']
    )
    .build()
)

# Resultado:
print(param_grid)


# strategy para criar classificadores:
from abc import ABC, abstractmethod

class GridStrategy(ABC):
    @abstractmethod
    def build_grid(self, builder: 'ParamGridBuilder') -> list:
        pass

class LogisticRegressionStrategy(GridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(LogisticRegression(), C=[0.1, 1, 10], penalty=['l1', 'l2'])
            .build()
        )

class RandomForestStrategy(GridStrategy):
    def build_grid(self, builder):
        return (
            builder
            .add_classifier(RandomForestClassifier(), n_estimators=[50, 100], max_depth=[5, 10])
            .build()
        )

# Uso:
builder = ParamGridBuilder()
strategy = LogisticRegressionStrategy()
param_grid = strategy.build_grid(builder)  # Flexível: troque a estratégia em runtime






class Component():
    """
    The base Component interface defines operations that can be altered by
    decorators.
    """

    def operation(self) -> str:
        pass


class ConcreteComponent(Component):
    """
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    """

    def operation(self) -> str:
        return "ConcreteComponent"


class Decorator(Component):
    """
    The base Decorator class follows the same interface as the other components.
    The primary purpose of this class is to define the wrapping interface for
    all concrete decorators. The default implementation of the wrapping code
    might include a field for storing a wrapped component and the means to
    initialize it.
    """

    _component: Component = None

    def __init__(self, component: Component) -> None:
        self._component = component

    @property
    def component(self) -> Component:
        """
        The Decorator delegates all work to the wrapped component.
        """

        return self._component

    def operation(self) -> str:
        return self._component.operation()


class ConcreteDecoratorA(Decorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """

    def operation(self) -> str:
        """
        Decorators may call parent implementation of the operation, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        """
        return f"ConcreteDecoratorA({self.component.operation()})"


class ConcreteDecoratorB(Decorator):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def operation(self) -> str:
        return f"ConcreteDecoratorB({self.component.operation()})"


def client_code(component: Component) -> None:
    """
    The client code works with all objects using the Component interface. This
    way it can stay independent of the concrete classes of components it works
    with.
    """

    # ...

    print(f"RESULT: {component.operation()}", end="")

    # ...


if __name__ == "__main__":
    # This way the client code can support both simple components...
    simple = ConcreteComponent()
    print("Client: I've got a simple component:")
    client_code(simple)
    print("\n")

    # ...as well as decorated ones.
    #
    # Note how decorators can wrap not only simple components but the other
    # decorators as well.
    decorator1 = ConcreteDecoratorA(simple)
    decorator2 = ConcreteDecoratorB(decorator1)
    print("Client: Now I've got a decorated component:")
    client_code(decorator2)




    from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

class ParamGridComponent:
    """Interface base para os componentes do param_grid"""
    def get_params(self) -> list:
        pass

class BaseParamGrid(ParamGridComponent):
    """Implementação concreta básica do param_grid"""
    def get_params(self) -> list:
        return []

class ParamGridDecorator(ParamGridComponent):
    """Classe base para todos os decoradores"""
    _component: ParamGridComponent = None

    def __init__(self, component: ParamGridComponent) -> None:
        self._component = component

    @property
    def component(self) -> ParamGridComponent:
        return self._component

    def get_params(self) -> list:
        return self._component.get_params()

class LogisticRegressionDecorator(ParamGridDecorator):
    """Adiciona configurações de LogisticRegression ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [LogisticRegression()],
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear']
        })
        return params

class RandomForestDecorator(ParamGridDecorator):
    """Adiciona configurações de RandomForest ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [RandomForestClassifier()],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10]
        })
        return params

class SVCDecorator(ParamGridDecorator):
    """Adiciona configurações de SVC ao param_grid"""
    def get_params(self) -> list:
        params = self.component.get_params()
        params.append({
            'classifier': [SVC()],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__C': [0.1, 1, 10]
        })
        return params

class CommonParamsDecorator(ParamGridDecorator):
    """Adiciona parâmetros comuns a todos os classificadores"""
    def __init__(self, component: ParamGridComponent, random_states=None):
        super().__init__(component)
        self.random_states = random_states or [42]

    def get_params(self) -> list:
        params = self.component.get_params()
        for config in params:
            config['classifier__random_state'] = self.random_states
        return params

# Código cliente
if __name__ == "__main__":
    # Criando o param_grid básico
    base_grid = BaseParamGrid()
    
    # Decorando com vários classificadores
    decorated_grid = CommonParamsDecorator(
        SVCDecorator(
            RandomForestDecorator(
                LogisticRegressionDecorator(base_grid)
            )
        ),
        random_states=[42, 101]
    )
    
    # Obtendo o param_grid final
    param_grid = decorated_grid.get_params()
    
    # Mostrando o resultado
    print("ParamGrid construído:")
    for i, config in enumerate(param_grid, 1):
        print(f"\nConfiguração {i}:")
        for key, value in config.items():
            print(f"{key}: {value}")





            from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Dados de exemplo
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir param_grid com múltiplos classificadores
param_grid = [
    {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
]

# Executar GridSearchCV
grid_search = GridSearchCV(
    estimator=LogisticRegression(),  # Isso será sobrescrito
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    refit=True,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Obter os melhores de cada classificador
results = pd.DataFrame(grid_search.cv_results_)
results['classifier_name'] = results['param_classifier'].apply(lambda x: x.__class__.__name__)
best_per_classifier = results.loc[results.groupby('classifier_name')['mean_test_score'].idxmax()]

# Mostrar resultados
print(best_per_classifier[['classifier_name', 'mean_test_score', 'params']])




# Estágio 1: Ambiente de construção (build) - Instala todas as dependências necessárias
FROM python:3.9-slim as builder

WORKDIR /app

# Instala dependências de sistema necessárias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Cria e ativa um ambiente virtual Python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scipy pandas scikit-learn statsmodels

# Copia todo o código fonte
COPY . .

# Estágio 2: Ambiente de treinamento - Executa o treinamento do modelo
FROM python:3.9-slim as trainer

WORKDIR /app

# Copia apenas o ambiente virtual do estágio builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia os scripts necessários para o treinamento
COPY train_model.py .
COPY data_preprocessing.py .
COPY config.yaml .

# Comando para treinar o modelo
CMD ["python", "train_model.py"]

# Estágio 3: Imagem final otimizada para produção
FROM python:3.9-slim as production

WORKDIR /app

# Copia apenas o necessário do estágio builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia apenas o modelo treinado e scripts de inferência
COPY --from=trainer /app/model.pkl .
COPY predict.py .
COPY requirements_prod.txt .

# Instala apenas as dependências necessárias para produção
RUN pip install --no-cache-dir -r requirements_prod.txt

# Comando para executar a API de predição (substitua conforme necessário)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "predict:app"]





from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd

# Supondo que você já tenha definido train_label e dev_label
label = pd.concat([train_label, dev_label], axis=0)

# Definir as métricas para avaliação
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

# Definir a validação cruzada estratificada (StratifiedKFold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definir o espaço de busca de hiperparâmetros (exemplo para LogisticRegression)
param_grid = {
    'C': [0.1, 1, 10],  # Se estiver usando LogisticRegression diretamente
    # Se estiver usando Pipeline, use 'logisticregression__C': [0.1, 1, 10]
}

# Criar o GridSearchCV
grid = GridSearchCV(
    estimator=LogisticRegression(),  # Substitua pelo seu modelo ou Pipeline
    param_grid=param_grid,
    cv=skf,
    scoring=scoring,
    refit='f1'  # Escolha a métrica principal para selecionar o melhor modelo
)

# Treinar o GridSearchCV (supondo que X e y já estão definidos)
# grid.fit(X, y)






import shap


mod = best_model.named_steps['classifier']
shap_values = explainer(hearth_test_preprocessed_todf)


# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(mod.predict_proba, predictors, link="logit")
shap_values = explainer.shap_values(hearth_test_preprocessed_todf, nsamples=100)


# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], 
                shap_values[0][0,:], 
                hearth_test_preprocessed_todf.iloc[0,:], link="logit")

# visualize the first prediction's explanation
shap.plots.bar(shap_values)






from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def download_kaggle_dataset(dataset_name, save_path):
    """
    Baixa um dataset do Kaggle
    
    Parâmetros:
    dataset_name (str): Nome no formato 'username/dataset-name'
    save_path (str): Pasta para salvar os arquivos
    """
    try:
        # Autenticar
        api = KaggleApi()
        api.authenticate()
        
        # Baixar dataset
        print(f"Baixando dataset {dataset_name}...")
        api.dataset_download_files(dataset_name, path=save_path, unzip=True)
        print("Download completo!")
        
        # Listar arquivos baixados
        import os
        print("\nArquivos baixados:")
        for file in os.listdir(save_path):
            print(f"- {file}")
            
    except Exception as e:
        print(f"Erro ao baixar dataset: {e}")

# Exemplo de uso
download_kaggle_dataset(
    dataset_name='heptapod/titanic',
    save_path='./titanic_data'
)

# Carregar os dados
train_df = pd.read_csv('./titanic_data/train.csv')
test_df = pd.read_csv('./titanic_data/test.csv')

print("\nDados carregados:")
print(f"Treino: {train_df.shape}")
print(f"Teste: {test_df.shape}")




# import sys
# import os
# # Adiciona o diretório acima ao PATH do Python
# sys.path.append(os.path.abspath(os.path.join('..')))


# from sampling.SplitData import SplitData
# splitter = SplitData(partitions = [.8,.1,.1])


# # heart_df.columns
# heart_df_label = heart_df.pop('label')


# train, train_label, dev, dev_label, test, test_label = \
#                                         splitter.get_three_sets(heart_df, 
#                                                                 heart_df_label)

# print(train.shape)
# print(dev.shape)
# print(test.shape)
# print(train.dtypes)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# import numpy as np


# ohe_obj = OneHotEncoder(sparse_output = False, 
#                         handle_unknown = 'ignore')

# # Lista das colunas categóricas
# train_cat = train.select_dtypes(include=['object'])
# cat_cols = train_cat.columns

# # Criar o transformador
# preprocessor = ColumnTransformer(
#     transformers = [('ohe', ohe_obj, cat_cols) ],
#     remainder = 'passthrough'  # Mantém as colunas não transformadas
# )


# df_encoded = pd.DataFrame(preprocessor.fit_transform(train), 
#                           columns = preprocessor.get_feature_names_out())

# # join com dados originais
# df_encoded = pd.concat([train, df_encoded], axis = 1)



# print(df_encoded.columns)
# print(df_encoded.shape)

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd
# import numpy as np


# ohe_obj = OneHotEncoder(sparse_output = False, 
#                         handle_unknown = 'ignore')

# # Lista das colunas categóricas
# train_cat = train.select_dtypes(include=['object'])
# cat_cols = train_cat.columns

# # Criar o transformador
# preprocessor = ColumnTransformer(
#     transformers = [('ohe', ohe_obj, cat_cols) ],
#     remainder = 'passthrough'  # Mantém as colunas não transformadas
# )


# df_encoded = pd.DataFrame(preprocessor.fit_transform(train), 
#                           columns = preprocessor.get_feature_names_out())

# # join com dados originais
# df_encoded = pd.concat([train, df_encoded], axis = 1)


# print(df_encoded.columns)
# print(df_encoded.shape)

# from sklearn.preprocessing import StandardScaler

# std_scaler_obj = StandardScaler()

# # Lista das colunas categóricas
# train_cat = train.select_dtypes(include=[np.number])
# num_cols = train_cat.columns

# # Criar o transformador
# preprocessor = ColumnTransformer(
#     transformers = [('stdscaler', std_scaler_obj, num_cols) ],
#     remainder = 'passthrough'  # Mantém as colunas não transformadas
# )


# df_encoded = pd.DataFrame(preprocessor.fit_transform(train), 
#                           columns = preprocessor.get_feature_names_out())

# print(df_encoded.shape)

# # join com dados originais
# df_encoded = pd.concat([train, df_encoded], axis = 1)

# print(df_encoded.columns)
# print(df_encoded.shape)
# print(train.shape)
,



mlflow



import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Configuração do MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # ou o URI do seu servidor MLflow
mlflow.set_experiment("Iris Classification - GridSearch")

# 2. Carregar e preparar os dados
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definir o modelo e os parâmetros para GridSearch
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 4. Configurar e executar o GridSearchCV com MLflow
with mlflow.start_run():
    # Iniciar GridSearch
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Log dos parâmetros e métricas
    mlflow.log_params(grid_search.best_params_)
    
    # Avaliar o melhor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    
    # Log do classification report como artefato
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv", index=True)
    mlflow.log_artifact("classification_report.csv")
    
    # Log do modelo
    mlflow.sklearn.log_model(best_model, "best_model")
    
    # Log de informações adicionais
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "Iris")
    
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    print(f"Acurácia do melhor modelo: {accuracy:.4f}")



