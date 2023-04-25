import pickle
import numpy as np
import streamlit as st
# Load the saved model from the pickle file and do predictions on the new dataset
loaded_model = pickle.load(
    open("C:/Users/Hp/Desktop/AI_Summative/final_model.pkl", "rb"))

# Function to predict the rating


def patient_test(input_data):

    input_data_numpy_array = np.array(input_data, dtype=object)
    input_data_reshaped = input_data_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    return prediction[0]


def main():
    st.title("Heart disease Prediction Application")

    # Get the features input from patient
    age = st.number_input("Enter the patient's ages data", min_value=0)
    sex = st.number_input("Enter the patient's data", min_value=0)
    chest_pain_type = st.number_input(
        "Enter the patient's data", min_value=0.00)
    cholesterol = st.number_input("Enter the patient's data", min_value=0.00)
    fasting_blood_sugar = st.number_input(
        "Enter the patient's data", min_value=0.00)

    resting_electrocardiogram = st.number_input(
        "Enter the patient's data", min_value=0.00)
    max_heart_rate_achieved = st.number_input(
        "Enter the patient's data", min_value=0)
    exercise_induced_angina = st.number_input(
        "Enter the patient's data", min_value=0)
    st_depression = st.number_input("Enter the patient's data", min_value=0, )
    st_slope = st.number_input("Enter the patient's data", min_value=0)
    num_major_vessels = st.number_input(
        "Enter the patient's data ", min_value=0)
    thalassemia = st.number_input("Enter the patient's data", min_value=0)

    rating = ""

    if st.button("Predict heart disease"):
        rating = patient_test([age, sex, cholesterol, chest_pain_type, fasting_blood_sugar, max_heart_rate_achieved,
                              resting_electrocardiogram, st_depression, exercise_induced_angina, st_slope, num_major_vessels, thalassemia])

    st.success(rating)


# if __name__ == '__main__':
#     try:
#         main()
#     except Exception as e:
#         st.write("Error: ", e)

if __name__ == '__main__':
    main()
