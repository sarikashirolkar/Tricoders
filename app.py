import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Career Prediction App")

# Load the trained LightGBM model using the new caching command for resources
@st.cache_resource
def load_model():
    model = lgb.Booster(model_file="model.txt")
    return model

model = load_model()

st.subheader("Enter Your Details")

# Create input fields for all features
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
education = st.selectbox("Highest Education Level", options=["High School", "Bachelor", "Master", "PhD"])
preferred_subjects = st.text_input("Preferred Subjects in Highschool/College")
academic_performance = st.number_input("Academic Performance (CGPA/Percentage)", min_value=0.0, max_value=10.0, value=7.0)
extracurricular = st.selectbox("Participation in Extracurricular Activities", options=["cultural","sports","debate","none"])
work_experience = st.selectbox("Previous Work Experience (If Any)", options=['Internship',"None", 'Part Time' ,'Full time'])
work_environment = st.selectbox("Preferred Work Environment", options=['Startup','Research','Corporate Job','Freelance'])
risk_taking = st.slider("Risk-Taking Ability", 1, 10, 5)
leadership = st.selectbox("Leadership Experience", options=['Student Council Member',"None" ,'Event Management'])
networking = st.selectbox("Networking & Social Skills", options=['Attended Corporate events',"None", 'Attended Buisness meets',
 'Attended Conferences'])
tech_savviness = st.slider("Tech-Savviness (1-10)", 1, 10, 5)
motivation = st.text_input("Motivation for Career Choice", "Enter your motivation here")

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Highest Education Level': [education],
    'Preferred Subjects in Highschool/College': [preferred_subjects],
    'academic_performance_cgpa': [academic_performance],
    'Participation in Extracurricular Activities': [extracurricular],
    'Previous Work Experience (If Any)': [work_experience],
    'Preferred Work Environment': [work_environment],
    'Risk-Taking Ability ': [risk_taking],
    'Leadership Experience': [leadership],
    'Networking & Social Skills': [networking],
    'Tech-Savviness': [tech_savviness],
    'Motivation for Career Choice ': [motivation]
})

# Function to perform label encoding for categorical features
def encode_input(df):
    # Dictionary of categorical columns and their corresponding classes
    # Adjust these lists to match the training phase!
    categorical_mappings = {
        'Gender': ["Male", "Female", "Other"],
        'Highest Education Level': ["High School", "Bachelor", "Master", "PhD"],
        'Preferred Subjects in Highschool/College': None,  # Free text field; use custom preprocessing if needed
        'Participation in Extracurricular Activities': ["Yes", "No"],
        'Previous Work Experience (If Any)': ["Yes", "No"],
        'Preferred Work Environment': ["Office", "Remote", "Hybrid"],
        'Leadership Experience': ["Yes", "No"],
        'Networking & Social Skills': ["Good", "Average", "Poor"],
        'Motivation for Career Choice ': None  # Free text field; use custom preprocessing if needed
    }
    
    # Process each categorical column if a mapping is provided
    for col, classes in categorical_mappings.items():
        if classes is not None:
            le = LabelEncoder()
            le.fit(classes)
            # Transform the input; if user input is not in the classes, this will raise an error
            df[col] = le.transform(df[col])
        else:
            # For free text fields, a more sophisticated preprocessing might be necessary.
            # For simplicity, we assign a default value (e.g., 0).
            df[col] = 0
    return df

# When the user clicks the "Predict Career" button, encode input and make prediction
if st.button("Predict Career"):
    encoded_input = encode_input(input_data.copy())
    
    # LightGBM model expects input in a specific format.
    prediction = model.predict(encoded_input)
    pred_class = np.argmax(prediction, axis=1)
    
    # Map numeric prediction to actual career labels (update with your mapping)
    career_map = {
        0: "Corporate Employee",
        1: "Entrepreneur",
        2: "Freelancer",
        3: "Govt Officer",
        4: "Research Scientist",
        5: "Not yet decided"    }
    st.write("Predicted Career:", career_map.get(pred_class[0], "Unknown"))