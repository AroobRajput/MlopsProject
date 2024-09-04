import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st


model = joblib.load("liveModelV1.pkl")

data = pd.read_csv('mobile_price_range.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]



X_train, X_test , y_train , y_test = train_test_split(X,y,testsize=0.2, random_state=)


y_pred = model.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Page Title
st.title("Model Accuracy and Real-Time Prediction")

# Display Accuracy
st.write(f"Model{accuracy}")

# Read time prediction 
st.header("Real-Time Prediction")
input_data = []
for col in X_test.columns:
    input_value = st.numner_input(f'Input for feature {col}', value='')
    input_data.append(input_value)

    #Convert input data to dataframe
input_df = pd.DataFrame([input_data], columns=X_test.columns)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f'Prediction:{prediction[0]}')