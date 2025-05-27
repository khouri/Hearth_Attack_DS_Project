

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

