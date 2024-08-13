from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import pandas as pd
from typing import Tuple


class Evaluation:
    def __init__(self, y_test, y_pred) -> None:
        self.y_test = y_test
        self.y_pred = y_pred

    # linear regression model training
    def model_evaluation(self) -> Tuple[float, float, float]:
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        logging.info(f'Mean Squared Error: {mse}')
        logging.info(f'Mean Absolute Error: {mae}')
        logging.info(f'R-squared: {r2}')

        return mse, mae, r2