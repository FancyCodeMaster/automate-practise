import logging
from zenml import step
import pandas as pd

from src.encoding import LabelEncoding

@step 
def encoding(dataframe:pd.DataFrame) -> pd.DataFrame:
    try:
        encode = LabelEncoding(dataframe=dataframe, categorical_column='Extracurricular Activities')
        dataframe = encode.encode_categorical()
        return dataframe
    except Exception as e:
        logging.error(f"Error in encoding : {e}")
        raise e