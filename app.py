import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('D:\\TSP4.0-AI-SDP-LAB-Repo-main\\shopping_trends_updated.csv')

# Preprocessing
data = data[['Category', 'Size', 'Payment Method', 'Location', 'Purchase Amount (USD)']]

# Encode categorical columns
encoder_category = LabelEncoder()
data['Category'] = encoder_category.fit_transform(data['Category'])

encoder_size = LabelEncoder()
data['Size'] = encoder_size.fit_transform(data['Size'])

encoder_payment = LabelEncoder()
data['Payment Method'] = encoder_payment.fit_transform(data['Payment Method'])

encoder_location = LabelEncoder()
data['Location'] = encoder_location.fit_transform(data['Location'])

# Split the dataset into features and target
X = data[['Category', 'Size', 'Payment Method', 'Location']]
y = data['Purchase Amount (USD)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Purchase Amount Predictor")
st.write("Predict the purchase amount based on the selected options.")

# User input
categories = encoder_category.classes_
sizes = encoder_size.classes_
payment_methods = encoder_payment.classes_
locations = encoder_location.classes_

selected_category = st.selectbox("Select a product category:", categories)
selected_size = st.selectbox("Select a size:", sizes)
selected_payment_method = st.selectbox("Select a payment method:", payment_methods)
selected_location = st.selectbox("Select a location:", locations)

# Prediction
if st.button("Predict"):
    category_encoded = encoder_category.transform([selected_category])[0]
    size_encoded = encoder_size.transform([selected_size])[0]
    payment_encoded = encoder_payment.transform([selected_payment_method])[0]
    location_encoded = encoder_location.transform([selected_location])[0]
    
    prediction = model.predict([[category_encoded, size_encoded, payment_encoded, location_encoded]])
    st.write(f"The predicted purchase amount for the selected options is ${prediction[0]:.2f}.")
