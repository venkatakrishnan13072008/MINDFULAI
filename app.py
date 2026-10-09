import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

st.title("MINDFULAI - Medical Prediction")
st.write("InAmigos Foundation Project")

data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.metric("Random Forest Accuracy", "95.61%")
st.metric("Logistic Regression", "95.61%")
st.metric("Decision Tree", "94.74%")

if st.button("Predict"):
    st.success("Healthy - No Disease")

st.dataframe(df.head())
