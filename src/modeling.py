from sklearn.linear_model import LinearRegression
import logging
import pandas as pd
import numpy as np


class Modeling:
    def __init__(self, X_train, y_train, X_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

    # linear regression model training
    def model_training(self) -> LinearRegression:
        reg = LinearRegression()

        model = reg.fit(self.X_train, self.y_train)

        score = model.score(self.X_train, self.y_train)

        logging.info(f'Model Score: {score}')

        return model
    
    # linear regression model prediction of X_test
    def model_inference(self, model) -> np.ndarray:
        y_pred = model.predict(self.X_test)
        logging.info('Predicted y value of X_test')
        return y_pred