import logging
from zenml import step
import pandas as pd
from typing import Tuple
from src.eval import Evaluation
import numpy as np

import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation(y_test:pd.Series, y_pred:np.ndarray) -> Tuple[float, float, float]:
    try:
        eval = Evaluation(y_test=y_test, y_pred=y_pred)
        mse, mae, r2 = eval.model_evaluation()
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        logging.info("Model Evaluation Completed")
        return mse,mae,r2
    except Exception as e:
        logging.error(f"Error in Evaluation : {e}")
        raise e