import dvc.api
import pandas as pd
import numpy as np

path = 'data/AdSmartABdata.csv'
repo = ' /home/Project/10 Acadamy Challenges/abtest-mlops'
version = 'v0'
data_url = dvc.api.get_url(
path = path,
repo = repo,
rev=version
)

if name == "main":
    warnings.filterwarnings("ignore")
    np.random.seed(60)

    data=pd.read_csv(data_url,sep=",")