import streamlit as st
import pickle
import numpy as np

# Function to load the saved model and encoders
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load the model and the encoders from the saved file
data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Function to display the prediction page
def show_predict_page():
    st.title("Farhan Software Developer Application")
    st.write("Hello! Hope you're doing well.")
    
    countries = (
        "Pakistan","United States", "India", "United Kingdom", "Germany", "Canada", 
        "Brazil", "France", "Spain", "Australia", "Netherlands", 
        "Poland", "Italy", "Russia", "Federation", "Sweden",
    )

    education = (
        "Less than a Bachelors", "Bachelor's degree", "Master's degree", "Post grad"
    )

    # User input fields
    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Button to trigger salary calculation
    ok = st.button("Calculate Salary")

    if ok:
        try:
            # Check the categories that the label encoder has been trained with
            st.write("Education Level Encoder Categories:", le_education.classes_)

            # Transform the inputs (country and education_level) using LabelEncoders
            # Try to safely handle the unseen labels scenario for both 'country' and 'education'
            if country not in le_country.classes_:
                st.error(f"Country '{country}' is not in the trained model's data. Please select a valid country.")
                return
            
            if education_level not in le_education.classes_:
                st.error(f"Education level '{education_level}' is not in the trained model's data. Please select a valid education level.")
                return
            
            # Transform country and education_level using LabelEncoders
            country_encoded = le_country.transform([country])[0]  # Transform country
            education_encoded = le_education.transform([education_level])[0]  # Transform education level
            
            # Prepare the input array for prediction (experience is already numeric)
            X = np.array([[country_encoded, education_encoded, experience]], dtype=float)
            
            # Predict salary using the model
            salary = regressor.predict(X)
            
            # Display the result
            st.subheader(f"The estimated salary is ${salary[0]:.2f}")
        except Exception as e:
            # Handle potential errors in transformation or prediction
            st.error(f"Error: {e}")
    
    # Optionally display the user input values
    st.write(f"Country Selected: {country}")
    st.write(f"Education Selected: {education_level}")
    st.write(f"Years of Experience: {experience}")

