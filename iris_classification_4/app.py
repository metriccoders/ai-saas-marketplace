import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

dataset = load_iris()

feature_names = dataset.feature_names

st.markdown("<span style='text-align: center; font-size: 44px;font-weight: bold;'>Iris Classification using One Class SVM - Metric Coders <span>", unsafe_allow_html=True)

st.header("Input Features")


sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.1, step=0.1)


features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

prediction = model.predict(features)

predicted_species = dataset.target_names[prediction][0]

st.markdown(f"<h3 style='color: green; text-align: center;'>The predicted species is: **{predicted_species}**</h3>", unsafe_allow_html=True)
