# 📈 Stock Prediction App

A **Streamlit app** to predict stock prices using an **LSTM (Long Short-Term Memory) model**.  
The app fetches historical stock data, trains a model, predicts trends, calculates RMSE, and forecasts the next 30 days of stock prices.

---

## Features

- Fetch and visualize historical stock data from Yahoo Finance  
- Train an LSTM model for stock trend prediction  
- Compare predictions with actual stock prices  
- Calculate **RMSE (Root Mean Squared Error)** for test predictions  
- Forecast next 30 days of stock prices  
- Interactive sidebar to select stock ticker, date range, and number of training epochs  

---

## Step-by-Step Instructions to Run the App

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YourUsername/stock-prediction-app.git
cd stock-prediction-app




2️⃣ Create a Virtual Environment (Recommended)
bash
Copy code
# Windows
python -m venv venv
venv\Scripts\activate




# Mac/Linux
python -m venv venv
source venv/bin/activate




3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt




4️⃣ Run the Streamlit App
bash
Copy code
streamlit run stock_prediction_app.py
Open the URL provided in the terminal (usually http://localhost:8501) in your browser.




5️⃣ Using the App
Enter Stock Ticker (e.g., AAPL, TSLA) in the sidebar

Select Start Date and End Date for historical data

Adjust Training Epochs using the slider

Scroll down to view:

Raw data table

Historical closing price chart

Train vs Test predictions

Test RMSE metric

30-day forecast plot





6️⃣ Optional: Explore Variables
You can print intermediate variables in Streamlit to understand the data and predictions:





RMSE Explanation
RMSE (Root Mean Squared Error) measures how far predictions are from actual prices

Smaller RMSE → more accurate predictions

Useful for evaluating model performance on test data

Requirements
Python 3.9+

Packages listed in requirements.txt

Author
Sanjaikumar

