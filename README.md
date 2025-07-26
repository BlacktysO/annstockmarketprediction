# annstockmarketprediction
This project demonstrates how an Artificial Neural Network (ANN) can be trained to predict the closing price of a stock based on historical Open, High, and Volume values. The dataset used for training is fetched from Yahoo Finance for the company Edelweiss Financial Services, and is processed and normalized before feeding into the model.

🧠 Project Objective
The goal of this project is to:

Predict the next day's closing price of a stock based on historical data.

Explore the use of a simple feedforward ANN model for regression tasks.

Analyze how well the ANN generalizes and how accurately it can predict unseen data.

📂 Dataset
Source: kaggle

File used: EDELWEISSNS.csv

Features used: Open, High, Volume

Target: Close

The original dataset also contains fields like Low, Adj Close, and date information (Year, Month, Day) which are dropped for simplicity.

🔧 Data Preprocessing
All prices (Open, High, Close) are normalized by dividing by 100.

Volume is scaled down initially (divided by 10,000), and then scaled back for uniformity.

Missing values are handled by replacing them with 0.

The dataset is split into 80% training and 20% testing.

🧮 Model Architecture
Implemented using Keras Sequential API:

Input Layer: 3 Neurons (Open, High, Volume)

Hidden Layer 1: Dense layer with 32 neurons and ReLU activation

Hidden Layer 2: Dense layer with 10 neurons and ReLU activation

Output Layer: Dense layer with 1 neuron (Linear activation for regression)

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Batch Size: 128
Epochs: 10
Validation Split: 5%

🧪 Model Evaluation
The model is trained on 1,973 samples and validated on 99 samples.

Evaluation Metrics:

Train Score: 0.01 MSE (0.10 RMSE)

Test Score: 1.24 MSE (1.12 RMSE)

These values indicate a well-fitted model with good generalization on unseen test data.

📊 Visualizations
The predicted closing prices from the ANN model are plotted against the real prices.

This helps visually validate how close the predictions are to the actual values.



✅ Key Highlights
Simple ANN model built from scratch using Keras.

Efficient data normalization and preprocessing.

Effective handling of missing data points.

Clear separation of training and testing sets to avoid overfitting.

Visually appealing and interpretable result graphs.

🚀 Future Improvements
Incorporate LSTM or GRU models to capture time-series dependencies.

Include additional features like technical indicators (SMA, EMA, RSI).

Perform hyperparameter tuning (layers, learning rate, activation functions).

Use MinMaxScaler or StandardScaler instead of manual normalization.

🛠 Technologies Used
Python

Pandas

NumPy

Matplotlib

Keras (TensorFlow backend)

📁 How to Run
Clone the repository.

Place the EDELWEISSNS.csv file in the root directory.

Install the required packages:

bash

pip install pandas numpy matplotlib keras

Run the script:

python stock_prediction_ann.py
