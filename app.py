from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol'].upper()

    try:
        df = yf.download(stock_symbol, period="1y", interval="1d")
        df = df.dropna()
        df['date_ordinal'] = pd.to_datetime(df.index).map(pd.Timestamp.toordinal)

        model = LinearRegression()
        model.fit(df['date_ordinal'].values.reshape(-1, 1), df['Close'].values)

        # future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
        # future_ordinal = [date.toordinal() for date in future_dates]
        # predictions = model.predict(np.array(future_ordinal).reshape(-1, 1))

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
        future_ordinal = [date.toordinal() for date in future_dates]
        predictions = model.predict(np.array(future_ordinal).reshape(-1, 1)).flatten()

        prediction_text = f'Predicted prices for the next 5 days: {", ".join([f"${p:.2f}" for p in predictions])}'


        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df['Close'], label='Historical Prices', color='blue')
        plt.plot(future_dates, predictions, label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{stock_symbol} Stock Price Prediction')
        plt.legend()
        plt.tight_layout()

        # Save to memory (not disk)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        prediction_text = f'Predicted prices for the next 5 days: {", ".join([f"${p:.2f}" for p in predictions])}'
        return render_template('index.html', prediction_text=prediction_text, plot_url=plot_url)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
