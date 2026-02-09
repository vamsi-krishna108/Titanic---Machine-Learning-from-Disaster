import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ---------- DATA & MODEL ----------
@st.cache_data
def load_data():
    # CSV file must be in the same folder as app.py
    data = pd.read_csv("Titanic - Machine Learning from Disaster.csv")
    # Same preprocessing as notebook
    data.drop(columns=["Cabin", "Age", "Embarked", "Ticket"], inplace=True)
    X = data.drop(columns=["Survived", "Name"])
    y = data["Survived"]
    X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
    return X, y

@st.cache_resource
def train_model():
    X, y = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2
    )

    # Tuned SVC from notebook (C=1, kernel='rbf', gamma='scale')
    model = SVC(C=1, kernel="rbf", gamma="scale", random_state=2)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, train_acc, test_acc

model, scaler, train_acc, test_acc = train_model()

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">🚢 Titanic Survival Predictor</h1>
    <p style="text-align:center; color:gray; margin-top:4px;">
        SVC model (~77% test accuracy) trained from your Titanic notebook.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------- METRICS ----------
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Train Accuracy", f"{train_acc:.1%}")
with col_b:
    st.metric("Test Accuracy", f"{test_acc:.1%}")

st.markdown("---")

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=0)
sex = st.sidebar.selectbox("Sex", ["male", "female"], index=0)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard")