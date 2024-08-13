from pipelines.train_pipeline import training_pipeline
import pipeline_config

from zenml.client import Client
import mlflow


if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(
        rootfolder=pipeline_config.ROOT_PATH,
        dataset=pipeline_config.datasetfolder,
        csvfilename=pipeline_config.csvfilename,
        dependent_variable=pipeline_config.dependent_variable,
        test_size=pipeline_config.test_size
    )


# from zenml.client import Client

# def main():
#     client = Client()
#     if client.active_stack.experiment_tracker:
#         tracking_uri = client.active_stack.experiment_tracker.get_tracking_uri()
#         print(f"Tracking URI: {tracking_uri}")
#     else:
#         print("Experiment tracker is not configured in the active stack.")

# if __name__ == "__main__":
#     main()

