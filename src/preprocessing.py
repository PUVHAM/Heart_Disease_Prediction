import numpy as np

def preprocess_input(input_data):
    # Convert categorical variables to numerical
    input_data['sex'] = 1 if input_data['sex'] == 'Male' else 0
    input_data['cp'] = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(input_data['cp'])
    input_data['fbs'] = 1 if input_data['fbs'] == 'Yes' else 0
    input_data['restecg'] = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(input_data['restecg'])
    input_data['exang'] = 1 if input_data['exang'] == 'Yes' else 0
    input_data['slope'] = ['Upsloping', 'Flat', 'Downsloping'].index(input_data['slope'])
    input_data['thal'] = [3, 6, 7][['Normal', 'Fixed Defect', 'Reversible Defect'].index(input_data['thal'])]
    
    # Convert to numpy array
    return np.array(list(input_data.values())).reshape(1, -1)
