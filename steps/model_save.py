import logging
from zenml import step
from src.model_save import ModelSave
from src.modeling import Modeling


@step()
def model_save(deployment_decision:bool, model:Modeling, model_save_path:str) -> bool: 
    try:
        if deployment_decision:
            ms = ModelSave(save_path=model_save_path, model=model)
            ms.save_model()
            logging.info(f'Model saved to path : {model_save_path}')
            return True
        else:
            logging.info('Model not saved due to less accuracy')
            return False
    except Exception as e:
        logging.error(f"Error in model save : {e}")
        raise e
