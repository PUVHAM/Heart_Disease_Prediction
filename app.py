import joblib
import streamlit as st
from src.train import train_models_parallel, train_and_cache_models
from src.data_analysis import plot_figure
from src.preprocessing import preprocess_input
from src.config import ModelConfig
from src.utils import get_feature

def main():
    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon=":heart:",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/yourusername/heart_disease_prediction',
            'Report a Bug': 'mailto:youremail@example.com',
            'About': "# Heart Disease Prediction App\n"
                     "Predict heart disease using multiple machine learning models."
        }
    )

    st.title(':heart: Heart Disease Prediction App with :blue[Machine Learning]')
    
    st.markdown("""
    Welcome to the **Heart Disease Prediction App**! This app allows you to predict the likelihood of heart disease 
    using multiple machine learning models. Feel free to explore different models and see how they perform on heart disease prediction tasks.
    """)

    with st.sidebar:
        st.header("Configuration")
        
        option = st.selectbox('Choose Model Type', ModelConfig.MODEL_TYPES.keys())
        
        model_type = ModelConfig.MODEL_TYPES[option]
        
        show_analysis = st.toggle("Show Data Analysis")
        
        train_models = st.button('Train All Models')
        
        if train_models:
            with st.spinner('Training models...'):
                _, results = train_models_parallel(list(ModelConfig.MODEL_TYPES.values()))
            st.session_state.models_trained = True
            st.session_state.models_results = results
            if st.session_state.models_trained:
                st.success('Training completed!')

    # Input form for heart disease prediction
    st.subheader("Enter Patient Information")
    
    input_data = get_feature()

    predict_button = st.button('Predict', type="primary")

    if predict_button:
        if not st.session_state.get('models_trained', False):
            st.error("You need to train the models first before making predictions!")
        else:
            st.subheader("Heart Disease Prediction")
            model, result = train_and_cache_models(model_type)
            
            # Preprocess input data
            input_array = preprocess_input(input_data)
            
            scaler_path = 'src/data/scaler.pkl'
            scaler = joblib.load(scaler_path)
            input_array_scaled = scaler.transform(input_array)
            
            prediction = model.predict(input_array_scaled)
            
            st.write(f"**Model Used:** {result['model']}")
            st.write(f"**Prediction:** {'High risk of heart disease' if prediction[0] == 1 else 'Low risk of heart disease'} {'❗' if prediction[0] == 1 else '✅'}")
            st.write(f"**Model Train Accuracy:** {result['train_accuracy']}")
            st.write(f"**Model Test Accuracy:** {result['test_accuracy']}")

    if show_analysis:
        st.divider()
        plot_figure()

if __name__ == "__main__":
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        st.session_state.models_results = {}
        
    main()