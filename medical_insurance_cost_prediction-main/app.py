import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import streamlit as st
import time

# Load your data
data_path = 'C:/Users/saine/Downloads/p1/medical_insurance_cost_prediction-main/data/insurance.csv'
data = pd.read_csv(data_path)

# Preprocess the data
scaler = StandardScaler()
X = data.drop('charges', axis=1)
y = data['charges']

# Encoding categorical variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

X_scaled = scaler.fit_transform(X)

# Split data for training and testing (optional, but recommended)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the scaler
scaler_path = 'C:/Users/saine/Downloads/p1/medical_insurance_cost_prediction-main/new_scaler.pkl'
with open(scaler_path, 'wb') as file:
    pickle.dump(scaler, file)

# Save the model
model_path = 'C:/Users/saine/Downloads/p1/medical_insurance_cost_prediction-main/new_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

# Streamlit app code

@st.cache_resource
def model_load(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def transformation_load(path):
    with open(path, 'rb') as file:
        transformation = pickle.load(file)
    return transformation

# Load the scaler and model
scaler = transformation_load(scaler_path)
model = model_load(model_path)

# Streamlit app code
image_path = 'C:/Users/saine/Downloads/p1/medical_insurance_cost_prediction-main/istockphoto-868640146-1024x1024.jpg'
st.image(image_path, width=400)

st.title('Medical Insurance Cost Predictor')
st.markdown('#### This model can predict medical charges with an accuracy score of 90%')

# Input fields
st.markdown("#### Age")
age = st.text_input('Age: ')

st.markdown('#### Gender')
gender = st.selectbox("Select Gender", ["Male", "Female"])

st.markdown("#### BMI")
bmi = st.text_input("Enter BMI value in range of (15-55)")

st.markdown("#### Number of Children")
children = st.text_input("Input number of children")

st.markdown("#### Smoker")
smoker = st.selectbox("Smoker", ["Yes", "No"])

st.markdown("#### Region")
region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

if st.button('Predict'):
    try:
        # Encoding the input data similar to the training data
        gender_encoded = 1 if gender == "Male" else 0
        smoker_encoded = 1 if smoker == "Yes" else 0
        region_encoded = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}[region]

        data = [int(age), float(bmi), int(children), gender_encoded, smoker_encoded,
                region_encoded == 1, region_encoded == 2, region_encoded == 3]

        scaled_data = scaler.transform([data])
    except ValueError:
        st.markdown("### Please enter valid data!")
    else:
        result = model.predict(scaled_data)

        bar = st.progress(50)
        time.sleep(1)
        bar.progress(100)

        st.info('Success')
        st.markdown(f'**Your Predicted Health Insurance Charge is: $ {result[0]:.2f}**')

# Instructions to Run the App
# 1. Make sure you have all required packages installed:
#    pip install streamlit scikit-learn pandas
# 2. Run the Streamlit app using the command:
#    streamlit run C:/Users/saine/Downloads/p1/medical_insurance_cost_prediction-main/app.py
