# Stock-Price-Prediction-Using-LSTM
This project demonstrates a machine learning workflow for predicting stock prices using Long Short-Term Memory (LSTM) networks in PyTorch. The dataset used contains historical stock prices, and the focus is on preparing, training, and evaluating the LSTM model.
### Overview
This notebook includes:
1. Loading and exploring stock price data.
2. Preprocessing and feature scaling for time series modeling.
3. Building an LSTM-based deep learning model.
4. Training the model and evaluating its performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
### Dataset
The dataset used for this project is:
1. Google_Stock_Price_Train.csv: Contains historical Google stock prices.
### Prerequisites
- Python 3.7+

**Required Libraries:**
- pandas
- numpy
- matplotlib
- torch
- scikit-learn

### Notebook Workflow
**Step 1: Data Loading**
- The dataset is loaded using pandas.
- Basic exploratory analysis is performed to check for null values and understand its structure.
  
**Step 2: Data Preprocessing**
- Feature scaling using MinMaxScaler.
- Splitting the data into training and testing sets.
  
**Step 3: Model Development**
- An LSTM model is built using PyTorch.
- The model is trained using the Adam optimizer and MSE loss function.
  
**Step 4: Evaluation**
The model's performance is evaluated using:
- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
- Root Mean Squared Error (RMSE): Provides a measure of error magnitude.
- Mean Absolute Error (MAE): Evaluates the average magnitude of errors in predictions.

Predictions are visualized for comparison with actual stock prices.

### Results
The LSTM model successfully captures the trends in stock prices. Future enhancements could involve:
- Incorporating external features to improve prediction accuracy.


