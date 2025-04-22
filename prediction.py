
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes_data.csv')
    data['Risk_Score'] = (data['hbA1c_level'] * 0.6) + (data['blood_glucose_level'] * 0.4)
    data = data.drop(['year', 'location', 'race:AfricanAmerican', 'race:Asian', 'race:Caucasian', 'race:Hispanic', 'race:Other'], axis=1)
    return data

data = load_data()

# Splitting Data
X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('scaler', MinMaxScaler(), ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'Risk_Score']),
    ('encoder', OneHotEncoder(), ['gender', 'smoking_history'])
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),  
    ('classifier', XGBClassifier(objective="binary:logistic", 
                                 subsample=0.8, reg_lambda=0.1, reg_alpha=0,
                                 n_estimators=250, max_depth=5, learning_rate=0.01,
                                 gamma=0.1, colsample_bytree=0.9))
])

# Train Model
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#accuracy = accuracy_score(y_test, y_pred)
#print(f"### Model Test Accuracy: {accuracy:.4f}")
#print("Classification Report:")
#print(classification_report(y_test, y_pred))

'''# Display Evaluation
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Test Accuracy: {accuracy:.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))'''

# Streamlit UI
st.title("Diabetes Prediction App 🩺")
st.write("Enter your health parameters to check your diabetes risk.")

# User Input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
hbA1c_level = st.number_input("HbA1c Level", min_value=0.0, format="%.2f")
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0)
smoking_history = st.selectbox("Smoking History", ["never", "Former", "Current"])
hypertension = st.selectbox("Hypertension", [0, 1])  # 0 for No, 1 for Yes
heart_disease = st.selectbox("Heart Disease", [0, 1])

risk_score = (hbA1c_level * 0.6) + (blood_glucose_level * 0.4)


# Prediction
if st.button("Predict"):
    input_data = np.array([gender, age, hypertension, heart_disease, smoking_history, bmi, hbA1c_level, blood_glucose_level,risk_score])
    column_names = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level','Risk_Score']

    # Convert to DataFrame
    input_data = pd.DataFrame([input_data], columns=column_names)

    # Now transform 
    input_transformed = preprocessor.transform(input_data)

    # Make Prediction
    prediction = pipeline.named_steps['classifier'].predict(input_transformed)


    if prediction[0] == 1:
        st.error("⚠️ High Risk: You may have diabetes.")
    else:
        if hbA1c_level > 6.4 or blood_glucose_level >= 140 or bmi >25:
            if 5.7 <= hbA1c_level <= 6.4:  # Corrected indentation and condition syntax
                st.warning("⚠️ You are Prediabetic. Higher risk of developing diabetes. Consider these lifestyle changes:")
                st.write("""
                - Maintain a **healthy diet** (reduce sugar and processed foods).  
                - Engage in **regular exercise** (30 minutes daily).  
                - Monitor your **blood sugar** regularly.  
                - Manage **stress levels** and get enough sleep.  
                - Visit your doctor for regular check-ups.  
                """)
            else:
                st.warning("⚠️ You are at risk of developing diabetes. Consider these lifestyle changes:")
                st.write("""
                - Maintain a **healthy diet** (reduce sugar and processed foods).  
                - Engage in **regular exercise** (30 minutes daily).  
                - Monitor your **blood sugar** regularly.  
                - Manage **stress levels** and get enough sleep.  
                - Visit your doctor for regular check-ups.  
            """)
        else:
            st.success("You are not diabetic! ")

