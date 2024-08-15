# Stock-Price-Prediction
Simple prediction of stock closing prices using ARIMA and LSTM. Run this on Google Colab using Python 3.0. The IPYNB is pretty self-explanatory, and contains a lot of information in the comments about how the code works.

## Project Overview

This project aims to predict the near-future closing prices of the top five technology stocks—Apple (AAPL), Microsoft (MSFT), Alphabet (GOOG), Amazon (AMZN), and Facebook (FB)—which dominate the S&P 500. Using historical market data, we explore and compare the performance of two prediction models: a traditional ARIMA model and an Artificial Neural Network (ANN) model with Long Short-Term Memory (LSTM) architecture.

## Objectives

- **Data Collection:** Retrieve historical stock data from Yahoo! Finance for the period 2015-2020 using the yfinance plugin.
- **Model Implementation:** Develop and test two predictive models:
  - ARIMA (AutoRegressive Integrated Moving Average) model.
  - LSTM (Long Short-Term Memory) neural network.
- **Performance Evaluation:** Compare the models based on key metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE). Additionally, evaluate computational times to assess the trade-offs between speed and accuracy.

## Data Collection

The yfinance plugin is used to obtain historical data for the following stocks:
- Apple (AAPL)
- Microsoft (MSFT)
- Alphabet (GOOG)
- Amazon (AMZN)
- Facebook (FB)

The data includes closing prices and daily returns for each stock from 2015 to 2020. This data is then preprocessed for analysis and model training.

## Methodology

### 1. Data Preprocessing

- **Normalization:** Stock price data is normalized to ensure consistency across the dataset.
- **Splitting Data:** The data is split into training, validation, and test sets to evaluate the model performance effectively.
- **Time Series Conversion:** The data is converted into a time series format suitable for the ARIMA and LSTM models.

### 2. Model Implementation

#### ARIMA Model
The ARIMA model is employed to predict stock prices based on historical patterns. The model is tuned to find the optimal parameters (p, d, q) that minimize the Akaike Information Criterion (AIC), ensuring the best fit for the data.

#### LSTM Model
An LSTM model is constructed using TensorFlow and Keras, which can handle sequential data efficiently. Hyperparameters are optimized using the Keras Tuner, and the model's performance is evaluated on both CPU and GPU runtimes to determine the benefits of hardware acceleration.

### 3. Performance Evaluation
The performance of both models is compared using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

Computational time for each model is also recorded to compare the trade-offs between accuracy and efficiency.

## Results

- **ARIMA Model:** Demonstrates a strong ability to predict stock prices with a relatively low computational cost. However, it may not capture complex patterns as effectively as neural networks.
- **LSTM Model:** Shows improved performance in capturing intricate patterns in the data, especially when using GPU acceleration. The model's accuracy is generally higher, but it requires more computational resources.

## Conclusion

This project highlights the power of Artificial Neural Networks, specifically LSTM, in predicting stock prices in the era of Big Data. While traditional models like ARIMA are still valuable for their simplicity and efficiency, neural networks offer superior accuracy at the cost of increased computational time.

## Requirements

To run the project, the following Python packages are required:

- yfinance
- keras-tuner
- pmdarima
- statsmodels
- scipy
- scikit-learn
- pandas
- numpy
- matplotlib
- tensorflow
- plotly

Install the required packages using the following commands:

```bash
pip install yfinance keras-tuner pmdarima statsmodels scipy scikit-learn pandas numpy matplotlib tensorflow plotly
```

## Future Work

- **Expand the Dataset:** Include more stocks or extend the time period to further validate model performance.
- **Incorporate Additional Features:** Consider incorporating more financial indicators or macroeconomic factors to improve model accuracy.
- **Explore Other Models:** Experiment with other machine learning models like Gradient Boosting or Support Vector Machines for stock price prediction.
