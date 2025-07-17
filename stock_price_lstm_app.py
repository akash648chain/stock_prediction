
# Stock Price Prediction for MSFT, AMZN, GOOG, IBM using LSTM with Streamlit UI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to load and preprocess data
def load_data(ticker, start_date='2012-01-01', end_date='2023-12-31'):
    df = yf.download(ticker, start=start_date, end=end_date)
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    time_step = 60
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler, close_prices

# Function to create and train LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to plot predictions
def plot_predictions(real, predicted, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real, label='Actual Price')
    ax.plot(predicted, label='Predicted Price')
    ax.set_title(title)
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.title("Stock Price Prediction using LSTM")
st.markdown("This app predicts historical stock prices using deep learning (LSTM).")

stocks = ['MSFT', 'AMZN', 'GOOG', 'IBM']
selected_stock = st.selectbox("Select Stock", stocks)

if st.button("Predict"):
    with st.spinner("Training the model. Please wait..."):
        X, y, scaler, real_prices = load_data(selected_stock)
        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=20, batch_size=64, verbose=0)

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

    st.success("Prediction completed!")
    plot_predictions(actual, predictions, f'{selected_stock} Stock Price Prediction')
