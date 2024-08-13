import logging
from zenml import step
import pandas as pd

from src.ingest_csv import INGEST_CSV

@step
def ingest_data(rootfolder, datasetfolder, csvfilename) -> pd.DataFrame:
    try:
        df = INGEST_CSV(rootfolder=rootfolder, datasetfolder=datasetfolder, csvfilename=csvfilename)
        dataframe = df.read_csv()
        logging.info(f'Dataframe shape  :{dataframe.head()}')
        logging.info(f'Dataframe type  :{type(dataframe)}')
        logging.info('Data Ingestion done successfully.')
        return dataframe
    except Exception as e:
        logging.error(f"Error in data ingestion : {e}")
        raise e