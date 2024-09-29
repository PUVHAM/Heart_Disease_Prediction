import streamlit as st
from src.config import FeatureConfig

def get_feature_input(feature_name, feature_type, description, units):
    label = f"{feature_name.capitalize()} ({description}){f' ({units})' if units else ''}"
    
    if feature_type == 'Integer':
        return st.number_input(label, value=0, step=1)
    
    elif feature_type == 'Float':
        return st.number_input(label, value=0.0, step=0.1)
    
    elif feature_type == 'Categorical':
        options = FeatureConfig.CATEGORICAL_FEATURES_OPTIONS.get(feature_name)
        if options:
            return st.selectbox(label, options)
        else:
            return st.text_input(label)
        
def get_feature():
    # Define feature information
    feature_info = [
        ('age', 'Integer', 'Age', 'years'),
        ('sex', 'Categorical', 'Sex', ''),
        ('cp', 'Categorical', 'Chest Pain Type', ''),
        ('trestbps', 'Integer', 'Resting Blood Pressure (on admission to the hospital)', 'mm Hg'),
        ('chol', 'Integer', 'Serum Cholesterol', 'mg/dl'),
        ('fbs', 'Categorical', 'Fasting Blood Sugar > 120 mg/dl', ''),
        ('restecg', 'Categorical', 'Resting ECG Results', ''),
        ('thalach', 'Integer', 'Maximum Heart Rate Achieved', ''),
        ('exang', 'Categorical', 'Exercise Induced Angina', ''),
        ('oldpeak', 'Float', 'ST Depression Induced by Exercise Relative to Rest', ''),  
        ('slope', 'Categorical', 'Slope of Peak Exercise ST Segment', ''),
        ('ca', 'Integer', 'Number of Major Vessels (0-3) Colored by Flourosopy', ''),
        ('thal', 'Categorical', 'Thalassemia', '')
    ]

    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, (feature_name, feature_type, description, units) in enumerate(feature_info):
        with col1 if i % 2 == 0 else col2:
            input_data[feature_name] = get_feature_input(feature_name, feature_type, description, units)
            
    return input_data
