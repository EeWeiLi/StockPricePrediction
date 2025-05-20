# Stock Price Prediction Using LSTM with Streamlit Interface

This project presents a deep learning-based system for predicting future stock prices using Long Short-Term Memory (LSTM) networks, coupled with an interactive user interface developed in Streamlit. The system enables users to select a stock, define a forecasting window, visualize historical price trends, and view forecasted future prices — all from a web-based dashboard.

## Motivation
Stock markets are volatile and affected by various unpredictable factors. Traditional models like ARIMA or linear regression often fall short when handling non-linear time series data. LSTM, a type of recurrent neural network (RNN), excels at capturing long-term dependencies and is well-suited for financial forecasting tasks. This project bridges the gap between model accuracy and user interpretability through an interactive, real-time dashboard.

## Features
- Interactive Streamlit interface
- Real-time stock selection and prediction horizon (30–100 days)
- Visualization of historical stock prices
- Forecasting of future closing prices using a trained LSTM model
- Export historical data as CSV
- Supports over 100 global stocks from a KaggleHub dataset
- Tuned LSTM model using Keras Tuner
- Comparison with ARIMA, RNN, SVR, and Random Forest benchmarks

## Technologies Used
- Python 3.10
- TensorFlow / Keras (LSTM, Dropout, BatchNorm)
- Streamlit (UI)
- KaggleHub (for automatic dataset fetch)
- Scikit-learn (Scaling, Metrics)
- Statsmodels (ARIMA)
- Pyngrok (for public link during Colab demo)

## Dataset
Dataset is pulled directly from Kaggle using:
"nelgiriyewithana/world-stock-prices-daily-updating"

- Includes: `Date`, `Ticker`, `Brand_Name`, `Open`, `High`, `Low`, `Close`, and `Volume`

## Model Evaluation
The LSTM model was evaluated using the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Benchmarked against:
- ARIMA
- Simple RNN
- Random Forest Regressor
- Support Vector Regressor (SVR)

### 1. Clone Repository
```bash
git clone https://github.com/EeWeiLi/FYP_StockPricePrediction_LSTM.git
cd FYP_StockPricePrediction_LSTM

## User Guide
1. Select a stock brand from the sidebar.
2. Choose the number of days to predict into the future (30 to 100).
3. Filter historical date range to visualize past trends.
4. Click "Show Predicted Values" to view a table of forecasted prices.
5. Download historical data as CSV if needed.
