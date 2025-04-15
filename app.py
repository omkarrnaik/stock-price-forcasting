# from flask import Flask, render_template, request
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import io
# import base64
# from sklearn.linear_model import LinearRegression

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     stock_symbol = request.form['stock_symbol'].upper()

#     try:
#         df = yf.download(stock_symbol, period="1y", interval="1d")
#         df = df.dropna()
#         df['date_ordinal'] = pd.to_datetime(df.index).map(pd.Timestamp.toordinal)

#         model = LinearRegression()
#         model.fit(df['date_ordinal'].values.reshape(-1, 1), df['Close'].values)

#         # future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
#         # future_ordinal = [date.toordinal() for date in future_dates]
#         # predictions = model.predict(np.array(future_ordinal).reshape(-1, 1))

#         future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
#         future_ordinal = [date.toordinal() for date in future_dates]
#         predictions = model.predict(np.array(future_ordinal).reshape(-1, 1)).flatten()

#         prediction_text = f'Predicted prices for the next 5 days: {", ".join([f"${p:.2f}" for p in predictions])}'

#         # Plotting
#         plt.figure(figsize=(10, 4))
#         plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')
#         plt.plot(future_dates, predictions, label='Predicted Prices', color='red')
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title(f'{stock_symbol} Stock Price Prediction')
#         plt.legend()
#         plt.tight_layout()
        

#         # Save to memory (not disk)
#         img = io.BytesIO()
#         plt.savefig(img, format='png')
#         img.seek(0)
#         plot_url = base64.b64encode(img.getvalue()).decode()

#         prediction_text = f'Predicted prices for the next 5 days: {", ".join([f"${p:.2f}" for p in predictions])}'
#         return render_template('index.html', prediction_text=prediction_text, plot_url=plot_url)

#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol'].upper()

    try:
        # Download data
        df = yf.download(stock_symbol, start="2020-01-01", end="2025-04-12")
        prices = df['Close'].dropna().values

        if len(prices) < 101:
            raise Exception("Not enough data to make predictions (need at least 101 days).")

        window_size = 100
        # X, y = [], []
        # for i in range(len(prices) - window_size):
        #     X.append(prices[i:i + window_size])
        #     y.append(prices[i + window_size])

        # X = np.array(X)
        # y = np.array(y)

        X, y = [], []
        for i in range(len(prices) - window_size):
            X.append(prices[i:i + window_size].flatten())  # ✅ FLATTENED
            y.append(prices[i + window_size])

        X = np.array(X)
        y = np.array(y)


        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next 5 days
        # future_predictions = []
        # last_window = prices[-window_size:]

        # Predict next 5 days
        future_predictions = []
        last_window = prices[-window_size:].flatten()  # ✅ Ensure 1D

        for _ in range(5):
            next_price = model.predict([last_window])[0]  # ✅ Wrap as 2D array
            future_predictions.append(next_price)
            last_window = np.append(last_window[1:], next_price)  # Shift window

        # Plotting
        dates = np.arange(len(prices))
        future_dates = np.arange(len(prices), len(prices) + 5)

        plt.figure(figsize=(10, 6))
        plt.plot(dates, prices, label='Actual Prices', color='blue')
        plt.plot(dates[window_size:], model.predict(X), label='Model Fit', color='red')
        plt.scatter(future_dates, future_predictions, label='Predicted Prices', color='green')
        plt.title(f'Random Forest 5-Day Forecast for {stock_symbol}')
        plt.xlabel('Time (Days)')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()

        # Save to memory
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        prediction_text = f'Predicted prices for the next 5 days: {", ".join([f"${p:.2f}" for p in future_predictions])}'

        return render_template('index.html', prediction_text=prediction_text, plot_url=plot_url)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

