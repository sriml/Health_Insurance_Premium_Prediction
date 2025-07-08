import pandas as pd
import joblib

model = joblib.load("artifacts/prem_pred_ex_model.joblib")
scaler = joblib.load("artifacts/prem_pred_ex_scaler.joblib")
cols = joblib.load("artifacts/prem_pred_ex_cols.joblib")

def calculate_total_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    return total_risk_score

def calculate_lifestyle_risk(physical_activity, stress_level):
    pa_scores = {
        'High': 0, 'Medium': 1, 'Low': 4
    }
    stress_scores = {
        'High': 4, 'Medium': 1, 'Low': 0
    }
    lifestyle_risk_score = pa_scores.get(physical_activity, 4) + stress_scores.get(stress_level, 0)

    return lifestyle_risk_score

def handle_scaling(df):
    cols_to_scale = scaler['cols_to_scale']
    scaler_ob = scaler['scaler']

    df['income_level'] = None
    df[cols_to_scale] = scaler_ob.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def preprocess_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = cols['cols']

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    #Manually assign values for each categorical input based on input_dict
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':  # Correct key usage with case sensitivity
            df['age'] = value
        elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
            df['income_lakhs'] = value


    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['total_risk_score'] = calculate_total_risk(input_dict['Medical History'])
    df['life_style_risk_score'] = calculate_lifestyle_risk(input_dict['Physical Activity'], input_dict['Stress Level'])
    df = handle_scaling(df)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    print("Prediction:", prediction)
    return int(prediction[0])
