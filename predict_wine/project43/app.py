import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_wine

model = joblib.load("wine_model.pkl")

dataset = load_wine()

feature_names = dataset.feature_names

st.markdown("<span style='text-align: center; font-size: 44px;font-weight: bold;'>Wine Classification using BernoulliNB - Metric Coders <span>", unsafe_allow_html=True)

st.header("Input Features")


alcohol = st.slider("alcohol", min_value=11.0, max_value=15.0, value=12.0, step=0.1)
malic_acid = st.slider("malic_acid", min_value=0.7, max_value=5.8, value=5.0, step=0.1)
ash = st.slider("ash", min_value=1.36, max_value=3.23, value=2.0, step=0.1)

alcalinity_of_ash = st.slider("alcalinity_of_ash", min_value=10.6, max_value=30.0, value=10.9, step=0.1)
magnesium = st.slider("magnesium", min_value=70.0, max_value=162.0, value=79.0, step=0.1)
total_phenols = st.slider("total_phenols", min_value=0.98, max_value=3.80, value=2.0, step=0.1)

flavanoids = st.slider("flavanoids", min_value=0.34, max_value=5.0, value=2.0, step=0.1)
nonflavanoid_phenols = st.slider("nonflavanoid_phenols", min_value=0.13, max_value=0.66, value=0.2, step=0.15)
proanthocyanins = st.slider("proanthocyanins", min_value=0.41, max_value=3.5, value=0.5, step=0.01)
color_intensity = st.slider("color_intensity", min_value=1.28, max_value=13.0, value=5.0, step=0.1)
hue = st.slider("hue", min_value=0.48, max_value=1.710, value=1.0, step=0.01)
od280_od315_of_diluted_wines = st.slider("od280/od280_od315_of_diluted_wines", min_value=1.27, max_value=4.0, value=2.0, step=0.1)
proline = st.slider("proline", min_value=278.0, max_value=1680.0, value=350.0, step=10.0)





features = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids,
                      nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline ]])

prediction = model.predict(features)

predicted_wine = dataset.target_names[prediction][0]

st.markdown(f"<h3 style='color: green; text-align: center;'>The predicted wine is: **{predicted_wine}**</h3>", unsafe_allow_html=True)
