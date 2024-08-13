import logging
from zenml import step
import pandas as pd

from src.split_dataset import SplitDataFrame
from typing import Tuple

@step
def split_dataset(dataframe:pd.DataFrame, dependent_variable:str, test_size:float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        split = SplitDataFrame(dataframe=dataframe, dependent_variable=dependent_variable, test_size=test_size)
        X_train, X_test, y_train, y_test = split.split_dataframe()
        logging.info('Data Split done successfully.')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data splitting : {e}")
        raise e