import os
import pandas as pd

class INGEST_CSV:
    def __init__(self, rootfolder, datasetfolder, csvfilename) -> None:
        self.rootfolder = rootfolder
        self.datasetfolder = datasetfolder
        self.csvfilename = csvfilename

    def read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.rootfolder,self.datasetfolder,self.csvfilename))
        return df
