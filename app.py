# Import necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as rt

# Title
st.title("Home Price Prediction using Machine Learning")

# Predictions
st.title("Please input your values below")

# Test model with new input from user
# Store values in a list
data = []

# Taking 13 different values
# Median Income value
medianIncome = st.number_input("Enter the Median Income: ")
data.append(medianIncome)

# Median Age value
medianAge = st.number_input("Enter the Median Age: ")
data.append(medianAge)

# Total number of rooms
totRooms = st.number_input("Enter the Total Number of Rooms: ")
data.append(totRooms)

# Total number of bedrooms
bRoom = st.number_input("Enter the Total Number of Bedrooms: ")
data.append(bRoom)

# Area population
population = st.number_input("Enter the Population: ")
data.append(population)

# Number of households in area
households = st.number_input("Enter the Households: ")
data.append(households)

# Latitude
lat = st.number_input("Enter the Latitude: ")
data.append(lat)

# Longitude
long = st.number_input("Enter the Longitude: ")
data.append(long)

# Distance to Coast
dtC = st.number_input("Enter the Distance to Coast: ")
data.append(dtC)

# Distance to LA
dtLA = st.number_input("Enter the Distance to LA: ")
data.append(dtLA)

# Distance to San Diego
dtSD = st.number_input("Enter the Distance to San Diego: ")
data.append(dtSD)

# Distance to San Jose
dtSJ = st.number_input("Enter the Distance to San Jose: ")
data.append(dtSJ)

# Distance to San Francisco
dtSF = st.number_input("Enter the Distance to San Francisco: ")
data.append(dtSF)

# Convert input to DataFrame
data = np.array(data).reshape(1, -1)
df = pd.DataFrame(data)

# Predictions
st.title("Model Options")

# Initialize session state to store results
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# Load ONNX models and make predictions
# Random Forest Model
rf_session = rt.InferenceSession("rf_r.onnx")
rf_input_name = rf_session.get_inputs()[0].name
random_forest_model_prediction = rf_session.run(None, {rf_input_name: df.to_numpy().astype(np.float32)})[0][0]
if st.button("Random Forest Model Prediction"):
    st.session_state.predictions["Decision Tree"] = (f"Predicted Median Value of House is: ${random_forest_model_prediction[0]:,.2f}")
    print("Decision Tree:", st.session_state.predictions["Decision Tree"])


# Decision Tree Model
dt_session = rt.InferenceSession("dt_r.onnx")
dt_input_name = dt_session.get_inputs()[0].name
decision_tree_model_prediction = dt_session.run(None, {dt_input_name: df.to_numpy().astype(np.float32)})[0][0]
if st.button("Decision Tree Model Prediction"):
    st.session_state.predictions["Decision Tree"] = (f"Predicted Median Value of House is: ${decision_tree_model_prediction[0]:,.2f}")
    print("Decision Tree:", st.session_state.predictions["Decision Tree"])


# MLP Model
mlp_session = rt.InferenceSession("mlp_r.onnx")
mlp_input_name = mlp_session.get_inputs()[0].name
mlp_prediction = mlp_session.run(None, {mlp_input_name: df.to_numpy().astype(np.float32)})[0][0]
if st.button("MLP Prediction"):
    st.session_state.predictions["Mutilayer Perceptron"] = (f"Predicted Median Value of House is: ${mlp_prediction[0]:,.2f}")
    print("Mutilayer Perceptron:", st.session_state.predictions["Mutilayer Perceptron"])

# Elastic Net Model
en_session = rt.InferenceSession("en_r.onnx")
en_input_name = en_session.get_inputs()[0].name
elastic_net_prediction = en_session.run(None, {en_input_name: df.to_numpy().astype(np.float32)})[0][0]
if st.button("Elastic Net Prediction"):
    st.session_state.predictions["Elastic Net"] = (f"Predicted Median Value of House is: ${elastic_net_prediction[0]:,.2f}")
    print("Elastic Net:", st.session_state.predictions["Elastic Net"])


# Linear Regression Model
lr_session = rt.InferenceSession("lr_r.onnx")
lr_input_name = lr_session.get_inputs()[0].name
linear_regression_prediction = lr_session.run(None, {lr_input_name: df.to_numpy().astype(np.float32)})[0][0]
if st.button("Multiple Linear Regression Prediction"):
    st.session_state.predictions["Multiple Linear Regression"] = (f"Predicted Median Value of House is: ${linear_regression_prediction[0]:,.2f}")
    print("Multiple Linear Regression:", st.session_state.predictions["Multiple Linear Regression"])

# Display all selected predictions
st.title("Model Predictions")
for model_name, result in st.session_state.predictions.items():
    st.write(f"{model_name}: {result}")


# Sample input
# 8.3, 41, 800, 129, 322, 126, 37.88,-122.23,9263.040773,556529.158342,735501.806984,67432.517001,21250.213767






















