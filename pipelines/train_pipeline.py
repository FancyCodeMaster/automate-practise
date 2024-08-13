from zenml import pipeline

# import all steps here
from steps.ingest_data import ingest_data
from steps.data_preparation import data_preparation
from steps.encoding import encoding
from steps.split_dataset import split_dataset
from steps.modeling import modeling
from steps.evaluation import evaluation


@pipeline
def training_pipeline(rootfolder, dataset, csvfilename, dependent_variable, test_size):
    dataframe = ingest_data(rootfolder=rootfolder, datasetfolder=dataset, csvfilename=csvfilename)
    dataframe = data_preparation(dataframe)
    dataframe = encoding(dataframe)
    X_train, X_test, y_train , y_test = split_dataset(dataframe=dataframe, dependent_variable=dependent_variable,test_size=test_size)
    model, y_pred = modeling(X_train=X_train, y_train=y_train, X_test=X_test)
    mse, mae, r2 = evaluation(y_test, y_pred)
