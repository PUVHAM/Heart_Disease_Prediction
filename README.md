# Heart Disease Prediction 

This project focuses on predicting the likelihood of heart disease in patients using various machine learning algorithms. The dataset used is the **Cleveland heart disease dataset** from the UCI Machine Learning Repository. A variety of classification algorithms are employed to predict whether a patient is at high risk of heart disease or not, based on multiple clinical features.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1YdsOnhpD7dW5z-95pooRuaktkviuNgRA" width="460" height="280" style="margin-right: 5%;" />
  <img src="https://drive.google.com/uc?export=view&id=15j48oG-3K5HT4jsObAdhL_efcG53RUk3" width="460" height="280"/>
</p>

You can try the live demo of the deployed application [here](https://heartdiseaseprediction-puvham.streamlit.app/).

## Table of Contents
- [Purpose](#purpose)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Usage](#usage)
  - [1. Setup_Conda_Environment](#1-setup-conda-environment)
  - [2. Run the Application](#2-run-the-application)
  - [3. Docker](#3-docker)
- [Dataset](#dataset)
- [Acknowledgments](#acknowledgments)

## Purpose

Heart disease is one of the leading causes of death worldwide, with ischemic heart disease and stroke being major contributors. Early detection and prevention are key in reducing fatalities. By analyzing large clinical datasets, machine learning techniques can be used to assist in the early diagnosis of heart-related conditions.

This project uses the **Cleveland heart disease dataset**, which includes 14 clinical features, to predict the likelihood of a heart disease diagnosis.

The dataset consists of 303 instances with the following attributes:

- **age**: Age in years
- **sex**: Gender of the patient (Male/Female)
- **cp**: Chest pain type (e.g., Typical Angina, Atypical Angina, Non-anginal Pain, Asymptomatic)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol (in mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (Yes/No)
- **restecg**: Resting electrocardiographic results (e.g., Normal, ST-T Wave Abnormality)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (Yes/No)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment (e.g., Upsloping, Flat, Downsloping)
- **ca**: Number of major vessels (0–3) colored by fluoroscopy
- **thal**: Thalassemia (Normal, Fixed Defect, Reversible Defect)
- **target**: Diagnosis of heart disease (0: no disease, 1: presence of disease)

## Project Structure

The project repository has the following structure:
- **src/data/**: Directory for storing data and scaler.
- **src/config.py**: Contains configuration options for dataset paths and model types.
- **src/data_analysis.py**: Contains functions to generate data visualizations.
- **src/load_dataset.py**: Functions to download and preprocess the dataset.
- **src/preprocessing.py**: Preprocessing pipeline for input data.
- **src/train.py**: Handles the model training and evaluation processes.
- **src/utils.py**: Helper functions for streamlit input
- **app.py**: Main script that runs the Streamlit application.
- **docker/Dockerfile**: Configuration for deploying the app via Docker.
- **requirements.txt**: List of project dependencies.

## Models Used

This project implements several machine learning algorithms, including:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **AdaBoost**
- **Gradient Boosting**
- **XGBoost**
- **Stacking Ensemble**

### Model Training and Evaluation

The project provides the capability to train these models and evaluate their performance using the Cleveland dataset. Each model's accuracy is measured on both the training and test data. 

A Stacking Ensemble model is also implemented, which combines multiple base models for better accuracy.

## Usage

### 1. Setup Conda Environment

#### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
  
1. **Clone the Repository:**
   
    ```bash
    git clone https://github.com/PUVHAM/Heart_Disease_Prediction.git
    cd Heart_Disease_Prediction
    ```

2. **Create and Activate Conda Environment:**

    ```bash
    conda create --name heart_disease_prediction python=3.11
    conda activate heart_disease_prediction
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 2. Run the Application
The app is built using Streamlit. You can start the app by running:
```bash
streamlit run app.py
```
This will launch a web interface where you can:
- Train models
- Input patient data to predict heart disease risk
- View model performance metrics
- Analyze data relationships (e.g., age vs heart disease)

### 3. Docker
#### Prerequisites
  - [Docker](https://www.docker.com/get-started): Make sure Docker is installed on your system.

To run the application in a Docker container, use the following commands:
```bash
docker build -t heart_disease_prediction -f docker/Dockerfile .
docker run -p 8501:8501 -it --name heart_disease_prediction heart_disease_prediction
```
Once the Docker container is running, open your web browser and go to:
```bash
http://localhost:8501
```
## Dataset
The dataset used in this project is the Cleveland Heart Disease Dataset available from the UCI Machine Learning Repository. The dataset can be automatically downloaded when running the app or manually via those link [(1)](https://archive.ics.uci.edu/dataset/45/heart+disease) or [(2)](https://drive.google.com/file/d/1oGsM9EAFWiE28jDXT1IKhWwsqzIOcXcN/view).

## Acknowledgments
- The dataset was sourced from the UCI Machine Learning Repository.
- Special thanks to the contributors and maintainers of the `scikit-learn`, `xgboost`, and `streamlit` libraries.

Feel free to reach out if you have any questions or issues!
