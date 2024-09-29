import time
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from tqdm import tqdm
from src.load_dataset import load_df, split_dataset
from src.config import DatasetConfig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

class HeartDiseaseModel:
    random_state = DatasetConfig.RANDOM_SEED
    
    def __init__(self, model_type='dt'):
        self.models = {
            'knn': KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'),
            'svm': SVC(C=1.0, kernel='rbf', random_state=self.random_state, gamma='scale'),
            'nb': GaussianNB(),
            'dt': DecisionTreeClassifier(criterion='gini', random_state=self.random_state, ccp_alpha=0.0, max_depth=10, min_samples_split=2),
            'rf': RandomForestClassifier(random_state=self.random_state, min_samples_leaf=1, max_features='sqrt', 
                                         criterion='gini', max_depth=10, min_samples_split=2, n_estimators=10),
            'ada': AdaBoostClassifier(random_state=self.random_state, learning_rate=1.0, n_estimators=50),
            'gb': GradientBoostingClassifier(random_state=self.random_state, learning_rate=0.1, n_estimators=100, 
                                             subsample=1.0, min_samples_split=2, max_depth=3),
            'xgb': XGBClassifier(objective="binary:logistic", random_state=self.random_state, n_estimators=100)
        }
        self.model = self.models.get(model_type)
                
    def train(self, x_train, y_train):
        print('Start training...')
        self.model.fit(x_train, y_train)
        print('Training completed!')
        
    def evaluate(self, x_train, y_train, x_test, y_test):
        y_pred = self.model.predict(x_test)
        cm_test = confusion_matrix(y_pred, y_test)
        y_pred_train = self.model.predict(x_train)
        cm_train = confusion_matrix(y_pred_train, y_train)
        
        accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
        accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
        
        return accuracy_for_train, accuracy_for_test
    
    def predict(self, input):
        input_array = np.array(input).reshape(1, -1)
        prediction = self.model.predict(input_array)
        return prediction
    
class StackingHeartDiseaseModel(HeartDiseaseModel):
    def __init__(self):
        super().__init__()
        
        self.base_models = [
            ('dt', self.models['dt']),
            ('knn', self.models['knn']),
            ('rf', self.models['rf']),
            ('gb', self.models['gb']),
            ('xgb', self.models['xgb']),
            ('ada', self.models['ada']),
            ('svm', self.models['svm']),
        ]
        
        self.model = StackingClassifier(estimators=self.base_models, final_estimator=self.models['xgb'])
        
@st.cache_data(max_entries=1000, ttl=3600)
def train_and_cache_models(model_type):
    df = load_df(DatasetConfig.DATASET_PATH)

    x_train, y_train, x_test, y_test = split_dataset(df)

    if model_type == 'stacking':
        model = StackingHeartDiseaseModel()
    else:
        model = HeartDiseaseModel(model_type)
    model.train(x_train, y_train)
    train_accuracy, test_accuracy = model.evaluate(x_train, y_train, x_test, y_test)
    
    results = {
        'model': model.model.__class__.__name__,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    return model, results

@st.cache_data(max_entries=1000)
def train_models_parallel(model_types):
    models = {}
    results = {}

    with ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(train_and_cache_models, model_type): model_type for model_type in model_types
        }

        for future in tqdm(as_completed(future_to_model), total=len(model_types), desc="Training models"):
            model_type = future_to_model[future]
            model, result = future.result()
            models[model_type] = model
            results[model_type] = result

    return models, results