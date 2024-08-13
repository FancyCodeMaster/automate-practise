import logging
import joblib


class ModelSave:
    def __init__(self, save_path, model) -> None:
        self.save_path = save_path
        self.model = model

    def save_model(self):
        joblib.dump(self.model, self.save_path)