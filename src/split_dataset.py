from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from typing import Tuple

class SplitDataFrame:
    def __init__(self, dataframe, dependent_variable, test_size) -> None:
        self.dataframe = dataframe
        self.dependent_variable = dependent_variable
        self.test_size = test_size

    # split dataframe
    def split_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        y = self.dataframe[self.dependent_variable]
        X = self.dataframe.drop(columns=[self.dependent_variable])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)
        logging.info('Dataframe splitted to train and test split successfully')
        return X_train, X_test, y_train, y_test

