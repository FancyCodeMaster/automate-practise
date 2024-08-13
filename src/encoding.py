from sklearn.preprocessing import LabelEncoder
import pandas as pd
import logging

class LabelEncoding:
    def __init__(self, dataframe, categorical_column) -> None:
        self.dataframe = dataframe
        self.categorical_column = categorical_column

    # encode categorical
    def encode_categorical(self) -> pd.DataFrame:
        encoder = LabelEncoder()
        self.dataframe[self.categorical_column] = encoder.fit_transform(self.dataframe[self.categorical_column])
        logging.info('Categorical Column Encoded')
        return self.dataframe