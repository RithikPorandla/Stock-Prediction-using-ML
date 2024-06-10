import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'A4HH9Q7I0L81M6G9'

# Define the stock symbol and the API endpoint
symbol = 'MSFT'  # Example: Apple Inc.
endpoint = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

# Make the API request and load the data into a DataFrame
response = requests.get(endpoint)
data_json = response.json()

# Extract relevant data from the response
prices = data_json['Time Series (Daily)']
data = pd.DataFrame(prices).T
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

# Keep only necessary columns
data = data[['4. close']]
data.columns = ['Close']

# Print the first few rows of the dataset
print("Stock Price Data:")
print(data.head())

# Data cleaning and preprocessing
# Handling missing values by forward filling or imputation
data['Close'].fillna(method='ffill', inplace=True)

# Convert 'Close' column to numeric
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop rows with NaN values after conversion
data.dropna(subset=['Close'], inplace=True)

# Feature Engineering
# Creating lag features to capture temporal patterns
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag7'] = data['Close'].shift(7)


# Feature Engineering
# Creating lag features to capture temporal patterns
data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag7'] = data['Close'].shift(7)

# Splitting the data into training and validation sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Hyperparameter tuning using a grid search
best_rmse = float('inf')
best_order = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            order = (p, d, q)
            model = ARIMA(train['Close'], order=order)
            model_fit = model.fit()

            predictions = model_fit.forecast(steps=len(test))
            mse = mean_squared_error(test['Close'], predictions)
            rmse = np.sqrt(mse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order

# Training the final model with the best hyperparameters
final_model = ARIMA(data['Close'], order=best_order)
final_model_fit = final_model.fit()

# Making predictions on future data
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]  # Predicting 30 days into the future
future_predictions = final_model_fit.forecast(steps=len(future_dates))

# Saving the final model for future use
final_model_fit.save('arima_model.pkl')

# Visualizing Predictions
plt.plot(data.index, data['Close'], label='Actual Prices')
plt.plot(test.index, predictions, label='Validation Predictions')
plt.plot(future_dates, future_predictions, label='Future Predictions')
plt.title('Stock Price Prediction with ARIMA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
