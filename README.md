<h1>Stock Prediction Model deployed on streamlit</h1>
A Stock Price Prediction Model based on that displays the 100 day moving average(100MA) and 200 day moving average(200MA) along with predicting the future model based off the stock price for the past 100 days.
I've used Keras to create the model in the Python notebook and used Streamlit for deployment alongside using the keras model to predict the stock price and Pandas-datareader and yfinance library to get the latest stock prices from Yahoo Finance.

<img width="936" alt="Screenshot 2024-04-04 at 11 52 21 AM" src="https://github.com/Shamitr-Mardikar/StockPrediction-Project/assets/73629015/b6df4b51-ad7b-410c-8ca2-0c617be1edbf">
 
The streamlit app takes the stock Ticker as user input and displays all the information alongside applying the prediction model on the chosen stock.

You can check out the deployed project on - https://stockpred-project.streamlit.app/
