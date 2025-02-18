import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to load and preprocess the data
@st.cache_data
def load_data():
    # Load the dataset (ensure the CSV file is in the same directory as your app)
    data = pd.read_csv("realtor-data.csv")
    # Correct column names for city and state based on CSV's actual columns
    # Assuming columns are named 'city' and 'state'; adjust if different
    data_processed = pd.get_dummies(data, columns=["city", "state"])
    return data, data_processed

# Load data
data, data_processed = load_data()

# Prepare the features and target variable
X = data_processed.drop("price", axis=1)
y = data_processed["price"]

# Function to train the linear regression model
@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Train the model
model = train_model(X, y)

# Streamlit UI
st.title("Real Estate Price Prediction App")
st.write("Enter the house details below to predict the price:")

# User inputs for numerical features
# Adjust the input keys to match the CSV's column names
bedrooms = st.number_input("Number of Bedrooms", min_value=0, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, value=2)
land_size = st.number_input("Total Land Size (in acres)", min_value=0.0, value=0.5, format="%.2f")
street_address_encoded = st.number_input("Street Address Encoded", min_value=0, value=100)

# For categorical features, use correct column names from the data
# Assuming columns are named 'city' and 'state'
cities = data["city"].unique()
states = data["state"].unique()
city_input = st.selectbox("city", options=sorted(cities))
state_input = st.selectbox("state", options=sorted(states))

# Create a DataFrame from the user input with correct column names
input_data = {
    "bedrooms": [bedrooms],  # Match CSV column name
    "bathrooms": [bathrooms],  # Match CSV column name
    "acre_lot": [land_size],  # Adjust to match CSV's land size column name
    "street_address_encoded": [street_address_encoded],  # Match CSV column name
    "city": [city_input],  # Match CSV column name
    "state": [state_input]  # Match CSV column name
}
input_df = pd.DataFrame(input_data)

# One-hot encode the input to match training features
input_df_processed = pd.get_dummies(input_df, columns=["city", "state"])
# Align columns with training data and fill missing with 0
input_df_processed = input_df_processed.reindex(columns=X.columns, fill_value=0)

# Predict when the button is clicked
if st.button("Predict House Price"):
    prediction = model.predict(input_df_processed)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")