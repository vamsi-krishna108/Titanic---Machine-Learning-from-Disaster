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
parch = st.sidebar.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.sidebar.slider("Ticket Fare", 0.0, 512.0, 7.25, step=0.25)

st.sidebar.info(
    "Tip: 1st class, female, higher fare, and some family on board "
    "usually increase survival odds according to the data."
)

# ---------- MAIN PREDICTION AREA ----------
st.subheader("Prediction")

st.write(
    "Set the passenger details in the sidebar and click the button below to see the prediction."
)

center_col = st.columns([1, 2, 1])[1]
with center_col:
    predict_btn = st.button("Predict Survival", use_container_width=True)

if predict_btn:
    # Build input row matching training features
    input_df = pd.DataFrame(
        {
            "PassengerId": [1],  # dummy
            "Pclass": [pclass],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Sex_male": [1 if sex == "male" else 0],
        }
    )

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    confidence = model.decision_function(input_scaled)[0]

    st.markdown("### Result")

    if pred == 1:
        st.success("✅ The model predicts: **Survived**")
    else:
        st.error("❌ The model predicts: **Did not survive**")

    st.caption(
        f"Model confidence (distance from decision boundary): `{abs(confidence):.2f}` "
        "(higher means more certain)."
    )

# ---------- EXPLANATION ----------
with st.expander("How this model works"):
    st.markdown(
        """
        - Trained on Kaggle's **Titanic - Machine Learning from Disaster** dataset.[file:1]  
        - Preprocessing:
          - Drop: `Cabin`, `Age`, `Embarked`, `Ticket`.[file:1]
          - Features: `PassengerId`, `Pclass`, `SibSp`, `Parch`, `Fare`, `Sex_male`.[file:1]
          - Standardization with `StandardScaler`.[file:1]
        - Model: Support Vector Classifier (SVC) with tuned hyperparameters:
          - `C = 1`, `kernel = 'rbf'`, `gamma = 'scale'`.[file:1]
        - Split: 80% train / 20% test with `random_state=2`.[file:1]
        """
    )
