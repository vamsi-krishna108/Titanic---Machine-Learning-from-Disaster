# Titanic---Machine-Learning-from-Disaster

***

# 🚢 Titanic Survival Predictor

Interactive Streamlit app that predicts whether a Titanic passenger would **survive** using an SVC model trained on the classic **Kaggle Titanic – Machine Learning from Disaster** dataset.

## 1. Project Overview

This app is a Python/Streamlit version of the Jupyter notebook `Titanic_Machine_Learning_from_Disaster.ipynb`.
It reproduces the same pipeline:

- Load Titanic CSV
- Drop unused columns (`Cabin`, `Age`, `Embarked`, `Ticket`)
- Encode `Sex` (create `Sex_male` dummy)
- Scale features with `StandardScaler`
- Train-test split (80/20, `random_state=2`)
- Train tuned **SVC** model (`C=1`, `kernel='rbf'`, `gamma='scale'`)
- Achieves ~**82%** train accuracy and ~**77.1%** test accuracy. 

The Streamlit UI lets you adjust passenger details and see whether the model predicts **Survived** or **Did not survive**.

## 2. Folder Structure

```text
your-project/
├── app.py
├── Titanic - Machine Learning from Disaster.csv
└── Titanic_Machine_Learning_from_Disaster.ipynb 
```

- `app.py` – main Streamlit app (real runnable code).
- CSV – Titanic dataset used both in notebook and app.

## 3. Setup & Installation

1. **Clone / download** this project into a folder.
2. Place the Titanic CSV file in the same folder as `app.py` and name it:

   ```text
   Titanic - Machine Learning from Disaster.csv
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   .venv\Scripts\activate       # Windows
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   or manually:

   ```bash
   pip install streamlit pandas numpy scikit-learn
   ``` 

## 4. Running the App (VSCode or Terminal)

From the project folder:

```bash
streamlit run app.py
```

- This will open the app in your browser at `http://localhost:8501`.
- If it does not open automatically, copy the URL from the terminal.

## 5. Using the App

1. Go to **Titanic Survival Predictor** section.
2. Adjust inputs:
   - Pclass (1, 2, 3)
   - Sex (male/female)
   - SibSp (siblings/spouses on board)
   - Parch (parents/children on board)
   - Fare (ticket price).

3. Click **Predict**.

The app will show:

- Model **Train Accuracy** and **Test Accuracy**.
- A message: **“Survived!”** or **“Did not survive.”**
- A confidence score derived from the SVC decision function.

## 6. Model Details

- Algorithm: **Support Vector Classifier (SVC)**
- Hyperparameters (from GridSearchCV in the notebook)
  - `C = 1`
  - `kernel = 'rbf'`
  - `gamma = 'scale'`
- Preprocessing:
  - Drop: `Cabin`, `Age`, `Embarked`, `Ticket`
  - Features: `PassengerId`, `Pclass`, `SibSp`, `Parch`, `Fare`, `Sex_male`
  - Scaling: `StandardScaler` on all features.

## 7. Troubleshooting

- **CSV not found error**  
  Make sure the file name is exactly:

  ```text
  Titanic - Machine Learning from Disaster.csv
  ```

- **Module not found**  
  Reinstall dependencies:

  ```bash
  pip install streamlit pandas numpy scikit-learn
  ```

- **App not loading**  
  Check terminal for errors, fix them, then rerun `streamlit run app.py`.

***
