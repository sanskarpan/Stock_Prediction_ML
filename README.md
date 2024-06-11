# Stock Market Predictor

This project predicts stock prices using a Long Short-Term Memory (LSTM) model. It leverages historical stock data and provides visualizations to compare predicted prices with actual prices.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Stock Market Predictor uses an LSTM model to forecast future stock prices based on historical data. The project includes a web interface built with Streamlit, allowing users to input a stock symbol, view historical data, and see predictions.

## Features

- Fetch historical stock data using yfinance.
- Split data into training and testing sets.
- Normalize data with MinMaxScaler.
- Load a pre-trained LSTM model for predictions.
- Visualize moving averages (50-day, 100-day, 200-day) with actual prices.
- Plot predicted prices against actual prices.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sanskarpan/Stock_Prediction_ML.git
    ```
2. Change to the project directory:
    ```bash
    cd Stock_Prediction_ML
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
2. Open your web browser and navigate to `http://localhost:8501`.
3. Enter a stock symbol (e.g., `GOOG`) and view the predictions and visualizations.

## Project Details

The core logic of the project is in the `app.py` file:

1. **Libraries and Model Loading**:
    - Import necessary libraries and load the pre-trained LSTM model.

2. **User Interface**:
    - Create a Streamlit interface for user input and data display.

3. **Data Collection and Preprocessing**:
    - Fetch historical stock data.
    - Split the data into training and testing sets.
    - Normalize the data.

4. **Model Prediction**:
    - Prepare the data for prediction.
    - Use the LSTM model to predict future stock prices.

5. **Visualization**:
    - Plot moving averages and predicted prices using Matplotlib.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

