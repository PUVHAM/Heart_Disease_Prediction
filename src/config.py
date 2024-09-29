class DatasetConfig:
    DATASET_DIR = 'src/data'
    SCALER_PATH = 'src/data/scaler.pkl'
    DATASET_PATH = 'src/data/cleveland.csv'
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    
class ModelConfig:
    MODEL_TYPES = {
        "K-Nearest Neighbors": "knn",
        "Support Vector Machine": "svm",
        "Naive Bayes": "nb",
        "Decision Tree": "dt",
        "Random Forest": "rf",
        "AdaBoost": "ada",
        "Gradient Boosting": "gb",
        "XGBoost": "xgb",
        "Stacking": "stacking"
    }
    
class FeatureConfig:
    CATEGORICAL_FEATURES_OPTIONS = {
        'sex': ['Male', 'Female'],
        'cp': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
        'fbs': ['No', 'Yes'],
        'restecg': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
        'exang': ['No', 'Yes'],
        'slope': ['Upsloping', 'Flat', 'Downsloping'],
        'thal': ['Normal', 'Fixed Defect', 'Reversible Defect']
    } 