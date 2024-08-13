## MLOPS USING ZENML, MLFLOW, DVC 

# INSTALLATIONS

# 1. Installing zenml
pip install zenml
pip install zenml["server"]

# 2. MLFLOW integrations with ZenML
zenml init
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml model-registry register mlflow_model_registry --flavor=mlflow
zenml stack register mlflow_stacks -a default -o default -d mlflow -e mlflow_tracker -r mlflow_model_registry --set

# starting mlflow application for tracking
print(Client().active_stack.experiment_tracker.get_tracking_uri()) # do this in train pipeline
--> copy the file uri 

mlflow ui --backend-store-uri 'copied uri'
e.g.
mlflow ui --backend-store-uri 'file:/home/fancycodemaster/.config/zenml/local_stores/d21c4833-530e-4f6a-b340-a557422498df/mlruns'

## Libraries for deployment pipeline
from zenml import pipeline, step
from zenml.config import DockerSettings

from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True, settings={"docker_settings":docker_settings})