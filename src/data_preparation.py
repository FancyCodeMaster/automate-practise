import pandas as pd
import logging

class DATA_PREPARATION:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    # handle missing data
    def handle_missing_data(self,dataframe) -> pd.DataFrame:
        # missing_columns = []
        # for column in self.dataframe.columns:
        #     missing_columns.append(self.dataframe.isna().value_counts()[column])
        #     # let's say this contains [False, False, False]

        # missing_columns = list(set(missing_columns))
        
        # if len(missing_columns) == 1 and missing_columns[0] == False:
        #     logging.info("No missing data; so no need")
        #     return self.dataframe
        # else:
        logging.info("No missing data; so no need")
        return dataframe
    
    # handle outlier values
    def handle_outlier_values(self,dataframe) -> pd.DataFrame:
        logging.info("No outlier values.")
        return dataframe
    
    def prepare_data(self) -> pd.DataFrame:
        # dataframe = self.handle_missing_data(self.dataframe)
        # dataframe = self.handle_outlier_values(dataframe)
        dataframe = dataframe.drop(['Extracurricular Activities'], axis=1)
        return self.dataframe