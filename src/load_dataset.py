import os
import gdown
import joblib
import pandas as pd

from tqdm import tqdm
from src.config import DatasetConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_dataset():
    file_id = '1oGsM9EAFWiE28jDXT1IKhWwsqzIOcXcN'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    os.makedirs(DatasetConfig.DATASET_DIR, exist_ok=True)
    
    gdown.download(url,
                   output=DatasetConfig.DATASET_PATH,
                   quiet=True,
                   fuzzy=True)
    
def split_dataset(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
     
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=DatasetConfig.TEST_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    os.makedirs(DatasetConfig.DATASET_DIR, exist_ok=True)
    with tqdm(total=1, desc="Saving Scaler", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        joblib.dump(scaler, DatasetConfig.SCALER_PATH)
        pbar.update(1)
    
    return x_train, y_train, x_test, y_test

def load_df(csv_path):
    if not os.path.exists(csv_path):
        try:
            download_dataset()
        except Exception as e:
            ERROR_MSG = 'Failed when attempting download the dataset. Please check the download process.'
            raise e(ERROR_MSG)
    df = pd.read_csv(csv_path, header = None)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    return df