import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Streamlit app title
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Load stock data
startdate = '2010-01-01'
enddate = '2024-06-28'
df = yf.download(user_input, start=startdate, end=enddate)

if df.empty:
    st.error('No data found for the given ticker symbol. Please check the symbol and try again.')
else:
    # Display data description
    st.subheader('Data from 2010-2024')
    st.write(df.describe())

    # Plot Closing Price vs Time
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    # Plot Closing Price vs Time with 100MA
    st.subheader('Closing Price vs Time Chart with 100MA')
    ma_100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma_100, 'r')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    # Plot Closing Price vs Time with 100MA and 200MA
    st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
    ma_100 = df.Close.rolling(100).mean()
    ma_200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma_100, 'r')
    plt.plot(ma_200, 'g')
    plt.plot(df.Close, 'b')
    st.pyplot(fig)

    # Prepare training and testing data
    train_data = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    test_data = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Check if train_data and test_data are not empty
    if train_data.empty or test_data.empty:
        st.error('Training or testing data is empty. Please check the data split.')
    else:
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_arr = scaler.fit_transform(train_data)

        # Load model
        try:
            model = load_model('keras_model.h5')
        except:
            st.error('Model file "keras_model.h5" not found. Please ensure it is in the correct path.')
            st.stop()

        # Prepare test data
        pst_100_ds = train_data.tail(100)
        final_df = pd.concat([pst_100_ds, test_data], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict
        y_predicted = model.predict(x_test)

        # Reverse scaling
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot Prediction vs Original
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)
