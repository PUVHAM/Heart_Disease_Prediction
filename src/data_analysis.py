import time
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DatasetConfig
from src.load_dataset import load_df

def visualize_age_heart_disease_relationship(df):
    # Distribution of target vs age
    sns.set_context("paper", font_scale=1, rc={"font.size": 3, "axes.titlesize": 15, "axes.labelsize": 10})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='age', hue='target', order=df['age'].sort_values().unique(), ax=ax)
    ax.set_xticks(np.arange(0, 80, 5))
    plt.title('Variation of Age for each target class')
    return fig
    
def visualize_age_sex_heart_disease_relationship(df):
    # barplot of age vs sex with hue = target
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, y='age', x='sex', hue='target', ax=ax)
    plt.title('Distribution of age vs sex with the target class')
    return fig

def plot_figure():
    df = load_df(DatasetConfig.DATASET_PATH)
    
    st.subheader("Data Analysis")
    with st.spinner('Processing...'):
        time.sleep(2)
        count_fig = visualize_age_heart_disease_relationship(df)
        bar_fig = visualize_age_sex_heart_disease_relationship(df)
    st.markdown("""
    In both charts of this analysis:
    - **target = 1**: Indicates that the person has heart disease
    - **target = 0**: Indicates that the person does not have heart disease
    """)
    
    cols = st.columns(2)
    
    lst_fig = [count_fig, bar_fig]
    
    for i, fig in enumerate(lst_fig):
        with cols[i]:
            st.pyplot(fig)