import streamlit as st
from prediction_helper import predict

st.title("Health Insurance Premium Prediction")

categorical_options = {
    'Gender': ['Male', 'Female'],
    'Marital Status': ['Unmarried', 'Married'],
    'BMI Category': ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Self-Employed', 'Freelancer'],
    'Region': ['Northwest', 'Northeast', 'Southwest', 'Southeast'],
    'Medical History': [
        'No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
        'Thyroid', 'Heart disease', 'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ],
    'Insurance Plan': ['Bronze', 'Silver', 'Gold'],
    'Physical Activity': ['High', 'Medium', 'Low'],
    'Stress Level': ['Low', 'Medium', 'High']
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)

# Assign inputs to the grid
with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100)
with row1[1]:
    number_of_dependants = st.number_input('Number of Dependants', min_value=0, step=1, max_value=5)
with row1[2]:
    income_lakhs = st.number_input('Income in Lakhs', step=1, min_value=0, max_value=200)

with row2[0]:
    insurance_plan = st.selectbox('Insurance Plan', categorical_options['Insurance Plan'])
with row2[1]:
    employment_status = st.selectbox('Employment Status', categorical_options['Employment Status'])
with row2[2]:
    region = st.selectbox('Region', categorical_options['Region'])

with row3[0]:
    gender = st.selectbox('Gender', categorical_options['Gender'])
with row3[1]:
    marital_status = st.selectbox('Marital Status', categorical_options['Marital Status'])
with row3[2]:
    smoking_status = st.selectbox('Smoking Status', categorical_options['Smoking Status'])

with row4[0]:
    medical_history = st.selectbox('Medical History', categorical_options['Medical History'])
with row4[1]:
    physical_activity = st.selectbox('Physical Activity', categorical_options['Physical Activity'])
with row4[2]:
    stress_level = st.selectbox('Stress Level', categorical_options['Stress Level'])

with row5[0]:
    bmi_category = st.selectbox('BMI Category', categorical_options['BMI Category'])

input_dict={
    'Age': age,
    'Number of Dependants': number_of_dependants,
    'Income in Lakhs': income_lakhs,
    'Insurance Plan': insurance_plan,
    'Employment Status': employment_status,
    'Gender': gender,
    'Marital Status': marital_status,
    'BMI Category': bmi_category,
    'Smoking Status': smoking_status,
    'Region': region,
    'Medical History': medical_history,
    'Physical Activity': physical_activity,
    'Stress Level': stress_level
}

if st.button("Predict"):
    print("Predict button clicked")
    prediction = predict(input_dict)
    st.success(f"The predicted premium is ${prediction}")