import streamlit as st
import pandas as pd
import huggingface_hub
from huggingface_hub import HfApi,hf_hub_download
import joblib
import os

# Connect to HuggingFace Space using token from git secret
print("Connecting to Huggingface...")
try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    print("Connected..")
except Exception as e:
    print(f"Error connecting to HuggingFace Space:{e}")

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="supravab/Tourism_Package_Prediction", filename="tourism_package_prediction_modelv1.joblib")

# Load the trained model
print("Loading tourism_package_prediction model from Huggingface...")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'tourism_package_prediction_modelv1.joblib' not found. Please train and save the model first.")
    model = None

# Streamlit UI for Tourism Package Prediction
print("Preparing Streamlit UI App for Tourism Package Prediction..")
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for the company, that predicts whether a customer purchase a tourist package.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (Age of the customer)", min_value=15, max_value=100, value=30)
Gender = st.selectbox("Gender (Gender of customer)", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus (Marital Status of customer)", ["Married", "Unmarried", "Divorced"])
Occupation = st.selectbox("Occupation (Occupation of customer)", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Designation = st.selectbox("Designation (Designation of customer)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
CityTier = st.selectbox("CityTier (city category based on living)", ["1", "2","3"])
MonthlyIncome = st.number_input("MonthlyIncome (customer’s monthly income)", min_value=0.0, value=50000.0)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Has Own Car?", ["Yes", "No"])

NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting (No of people accompanying the customer)", min_value=1, max_value=10, value=5)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (No of children accompanying the customer)", min_value=0, max_value=5, value=2)
NumberOfTrips = st.number_input("NumberOfTrips (No of trips per year)", min_value=0, max_value=10, value=3)

TypeofContact = st.selectbox("TypeofContact (Method by which customer was contacted)", ["Self Enquiry", "Company Invited"])
ProductPitched = st.selectbox("ProductPitched (Type of product pitched)", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
DurationOfPitch = st.number_input("DurationOfPitch (Duration of the sales pitch)", min_value=0, max_value=100, value=20)
NumberOfFollowups = st.number_input("NumberOfFollowups (Number of follow-ups by the salesperson)", min_value=0, max_value=10, value=2)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore (Pitch satisfaction score given by customer)", min_value=0, max_value=10, value=5)
PreferredPropertyStar = st.number_input("PreferredPropertyStar (Preferred rating given by customer)", min_value=1, max_value=5, value=2)

# Process Feature-engineered Columns
def AgeGroup(age):
    if age <= 18:
        return 'Young'
    elif 19 <= age <= 40:
        return 'Adult'
    else:
        return 'Old'

def IncomeCategory(income):
    if income < 20000:
        return 'Low'
    elif 20000 <= income <= 30000:
        return 'Mid'
    else:
        return 'High'

def PitchPeriodCategory(pitch):
    if pitch <= 10:
        return 'Short'
    elif 11 <= pitch <= 30:
        return 'Long'
    else:
        return 'High'


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Occupation': Occupation,
    'Designation': Designation,
    'CityTier': CityTier,
    'MonthlyIncome': MonthlyIncome,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'NumberOfTrips': NumberOfTrips,    
    'TypeofContact': TypeofContact,
    'ProductPitched': ProductPitched,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'PreferredPropertyStar': PreferredPropertyStar,
    # New derived features
    'HasChildren': 1 if NumberOfChildrenVisiting > 0 else 0,
    'AgeGroup': AgeGroup(Age),
    'IncomeCategory': IncomeCategory(MonthlyIncome),
    'PitchPeriodCategory': PitchPeriodCategory(DurationOfPitch)
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Purchase Yes" if prediction == 1 else "Purchase No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
