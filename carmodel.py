import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained pipeline and feature names
try:
    with open('car_price_model_pipeline2.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        all_feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Model or feature names file not found. Please ensure 'car_price_model_pipeline.pkl' and 'feature_names.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

st.set_page_config(page_title='Car Price Predictor')
st.title('Car Price Predictor')

# Extract unique values for categorical features from all_feature_names
# This assumes the format 'FeatureName_CategoryValue'
unique_brands = sorted(list(set([f.replace('Brand_', '') for f in all_feature_names if f.startswith('Brand_')]))) if any('Brand_' in s for s in all_feature_names) else ['Default Brand']
unique_fuel_types = sorted(list(set([f.replace('Fuel Type_', '') for f in all_feature_names if f.startswith('Fuel Type_')]))) if any('Fuel Type_' in s for s in all_feature_names) else ['Default Fuel Type']
unique_transmissions = sorted(list(set([f.replace('Transmission_', '') for f in all_feature_names if f.startswith('Transmission_')]))) if any('Transmission_' in s for s in all_feature_names) else ['Default Transmission']
unique_models = sorted(list(set([f.replace('Model_', '') for f in all_feature_names if f.startswith('Model_')]))) if any('Model_' in s for s in all_feature_names) else ['Default Model']

st.header('Enter Car Details:')

# Input widgets for car features
brand = st.selectbox('Brand', unique_brands)
year = st.number_input('Year', min_value=2000, max_value=2023, value=2015, step=1)
fuel_type = st.selectbox('Fuel Type', unique_fuel_types)
transmission = st.selectbox('Transmission', unique_transmissions)
mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=100000, step=1000)
model = st.selectbox('Model', unique_models)


if st.button('Predict Price'):
    # Create a DataFrame from user inputs, matching the columns used in training
    input_data = pd.DataFrame([{
        'Brand': brand,
        'Year': year,
        'Fuel Type': fuel_type,
        'Transmission': transmission,
        'Mileage': mileage,
        'Model': model
    }])

    try:
        # Make prediction
        predicted_price = pipeline.predict(input_data)[0]
        st.success(f'Predicted Car Price: ${predicted_price:,.2f}')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

        st.write("Please ensure all input fields are valid and the model can process them.")
