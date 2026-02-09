import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load data (matches your notebook)
@st.cache_data
def load_data():
    data = pd.read_csv('Titanic - Machine Learning from Disaster.csv')
    data.drop(columns=['Cabin', 'Age', 'Embarked', 'Ticket'], inplace=True)
    X = data.drop(columns=['Survived', 'Name'], axis=1)
    y = data['Survived']
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    return X, y

# Train model (replicates your notebook's SVC with best params)
@st.cache_resource
def train_model():
    X, y = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)
    
    model = SVC(C=1, gamma='scale', kernel='rbf', random_state=2)  # Best params from your notebook
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, scaler, train_acc, test_acc

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")
st.markdown("Interactive app based on your Jupyter notebook's SVC model (~77% test accuracy).")

model, scaler, train_acc, test_acc = train_model()

col1, col2 = st.columns(2)
with col1:
    st.metric("Train Accuracy", f"{train_acc:.1%}")
with col2:
    st.metric("Test Accuracy", f"{test_acc:.1%}")

st.subheader("Predict Survival")
pclass = st.selectbox("Pclass", [1, 2, 3])
name = st.text_input("Name (ignored in model)")
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.slider("SibSp", 0, 8, 0)
parch = st.slider("Parch", 0, 6, 0)
fare = st.slider("Fare", 0.0, 512.0, 7.25)

if st.button("Predict"):
    # Create input matching your features
    input_df = pd.DataFrame({
        'PassengerId': [1],  # Dummy
        'Pclass': [pclass],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [1 if sex == 'male' else 0]
    })
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    st.success("Survived!" if prediction == 1 else "Did not survive.")
    st.write(f"**Probability:** Survived: {prob[1]:.1%}, Not: {prob[0]:.1%}")

st.subheader("Model Insights")
st.write("Reproduces your notebook: Data prep, scaling, train/test split (80/20), tuned SVC (C=1, RBF kernel, gamma=scale).")[file:1]
