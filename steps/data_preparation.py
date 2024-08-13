import logging
from zenml import step
import pandas as pd

from src.data_preparation import DATA_PREPARATION

@step 
def data_preparation(dataframe:pd.DataFrame) -> pd.DataFrame:
    try:
        prep = DATA_PREPARATION(dataframe)
        dataframe = prep.prepare_data()
        logging.info("Data Preparation Completed")
        return dataframe
    except Exception as e:
        logging.info(f"Error during data preparation: {e}")
        raise e