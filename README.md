# Stock-Prediction-using-ML

This Python script utilizes the ARIMA (AutoRegressive Integrated Moving Average) model to predict stock prices for a given company, using historical daily stock price data obtained from the Alpha Vantage API.

### Dependencies:
- pandas
- numpy
- matplotlib
- statsmodels
- requests

### Description:
- **Data Retrieval**: Historical daily stock price data is fetched from the Alpha Vantage API for a specified company (in this case, Microsoft - 'MSFT').
- **Data Preprocessing**: The retrieved data is cleaned and preprocessed, including handling missing values, converting data types, and creating lag features to capture temporal patterns.
- **Splitting Data**: The dataset is split into training and validation sets for model evaluation.
- **Hyperparameter Tuning**: Grid search is employed to find the optimal parameters (p, d, q) for the ARIMA model by minimizing the root mean squared error (RMSE) on the validation set.
- **Model Training**: The final ARIMA model is trained using the entire dataset with the best hyperparameters obtained from the grid search.
- **Future Predictions**: The trained model is then used to predict stock prices for the next 30 days into the future.
- **Visualization**: Predictions are visualized alongside actual stock prices and validation predictions using matplotlib.

### How to Use:
1. Replace `'YOUR_API_KEY'` with your actual Alpha Vantage API key.
2. Specify the stock symbol of interest (e.g., `'MSFT'` for Microsoft).
3. Run the script to obtain predictions and visualize the results.
