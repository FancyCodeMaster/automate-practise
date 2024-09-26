import logging
from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.linear_model import LinearRegression


from src.modeling import Modeling
import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step()
def modeling(X_train:pd.DataFrame, y_train:pd.Series, X_test:pd.DataFrame) -> Tuple[Modeling, np.ndarray, LinearRegression]:
    try:
        mlflow.sklearn.autolog()
        model = Modeling(X_train=X_train, y_train=y_train, X_test=X_test)
        mod = model.model_training()
        y_pred = model.model_inference(mod)
        logging.info("Model trained and predicted for X_test")
        return model, y_pred, mod
    except Exception as e:
        logging.error(f"Error in modeling : {e}")
        raise e