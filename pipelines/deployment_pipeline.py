import json
import os

import numpy as np
import pandas as pd
from zenml import pipeline, step
# from zenml.config import DockerSettings
# from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.constants import MLFLOW, TENSORFLOW
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
import pandas as pd

from steps.ingest_data import ingest_data
from steps.data_preparation import data_preparation
from steps.encoding import encoding
from steps.split_dataset import split_dataset
from steps.modeling import modeling
from steps.evaluation import evaluation
from steps.model_save import model_save

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Set the MLflow tracking server URI to the remote server
os.environ["MLFLOW_TRACKING_URI"] = "http://103.94.159.8:8000"

# Set username and password for authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "ekghanti"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "removeitTy1*}0iSe4SGl#E-"

# Now your deployment will use the remote tracking server instead of local
print(f"MLflow server running at: {get_tracking_uri()}")

# docker_settings = DockerSettings(required_integrations=[MLFLOW])

# this is to trigger the decision to deploy model if accuracy is greateer than the base accuracy
@step
def deployment_trigger(
    accuracy: float,
    min_accuracy: float
) -> bool:
    return accuracy > min_accuracy

@pipeline#(enable_cache=True, settings={"docker" : docker_settings})
def continuous_deployment_pipeline(
    rootfolder, 
    dataset, 
    csvfilename, 
    dependent_variable, 
    test_size,
    min_accuracy,
    model_save_path,
    workers: int = 1,
    timeout: int = 100, #DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    dataframe = ingest_data(rootfolder=rootfolder, datasetfolder=dataset, csvfilename=csvfilename)
    dataframe = data_preparation(dataframe)
    dataframe = encoding(dataframe)
    X_train, X_test, y_train , y_test = split_dataset(dataframe=dataframe, dependent_variable=dependent_variable,test_size=test_size)
    mod, y_pred, regmodel = modeling(X_train=X_train, y_train=y_train, X_test=X_test)
    mse, mae, r2 = evaluation(y_test, y_pred)
    deployment_decision =  deployment_trigger(r2, min_accuracy)
    save_decision = model_save(deployment_decision, regmodel, model_save_path)
    # mlflow_model_deployer_step(
    #     model=model,
    #     deploy_decision=deployment_decision,
    #     workers=workers,
    #     timeout=timeout,
    # )
    # mlflow_register_model_step(
    #     model=model,
    #     name="student-performance",
    # )
