# implied_volatility_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime
from scipy.interpolate import griddata
from scipy.stats import norm
import plotly.graph_objects as go
import concurrent.futures

# Define functions
def get_option_name_and_settlement(coin):
    r = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option")
    result = json.loads(r.text)
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)
    return name, settlement_period

def fetch_option_data(option_name):
    r = requests.get(f'https://test.deribit.com/api/v2/public/get_order_book?instrument_name={option_name}')
    result = json.loads(r.text)
    return pd.json_normalize(result['result'])

def extract_details(instrument_name, coin):
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-[CP]", instrument_name)
    if match:
        expiration_date = match.group(1)
        strike_price = match.group(2)
        return expiration_date, strike_price
    return None, None

def get_option_data(coin, settlement_per):
    coin_name, settlement_period = get_option_name_and_settlement(coin)
    coin_name_filtered = [coin_name[i] for i in range(len(coin_name)) if settlement_period[i] == settlement_per]
    coin_df = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_option = {executor.submit(fetch_option_data, name): name for name in coin_name_filtered}
        for future in concurrent.futures.as_completed(future_to_option):
            try:
                data = future.result()
                data['settlement_period'] = settlement_per
                coin_df.append(data)
            except Exception as exc:
                print(f'Error fetching data: {exc}')

    if len(coin_df) > 0:
        coin_df = pd.concat(coin_df)

    columns = ['state', 'estimated_delivery_price']
    if not coin_df.empty:
        coin_df.drop(columns, inplace=True, axis=1)

    coin_df['Expiration Date'], coin_df['Strike Price'] = zip(*coin_df['instrument_name'].apply(lambda x: extract_details(x, coin)))
    today = datetime.today()
    coin_df['Time to Expiration'] = coin_df['Expiration Date'].apply(lambda x: (datetime.strptime(x, '%d%b%y') - today).days / 365 if x else None)

    return coin_df

# Black-Scholes model functions
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(market_price, S, K, T, r, initial_vol, option_type="call", tolerance=1e-5, max_iterations=100):
    sigma = initial_vol
    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega_value = vega(S, K, T, r, sigma)
        if vega_value < 1e-5:
            break
        price_difference = market_price - price
        sigma += price_difference / vega_value
        if abs(price_difference) < tolerance:
            return sigma
    return None

# Streamlit App
st.title("Implied Volatility Surface: Crypto Options")

# Sidebar - User Inputs
st.sidebar.header("Model Parameters")
coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH"])
settlement_period = st.sidebar.selectbox("Settlement Period", ["day", "week", "month"])
interest_rate = st.sidebar.slider("Interest Rate", 0.0, 0.1, 0.05)
min_strike_percent = st.sidebar.slider("Min Strike (% of Spot Price)", 0.5, 1.5, 0.73)
max_strike_percent = st.sidebar.slider("Max Strike (% of Spot Price)", 0.5, 1.5, 1.2)

# Fetch and display data
data = get_option_data(coin, settlement_period)

# Ensure that Strike Price and underlying_price columns are numeric
data['Strike Price'] = pd.to_numeric(data['Strike Price'], errors='coerce')
data['underlying_price'] = pd.to_numeric(data['underlying_price'], errors='coerce')

# Drop rows where Strike Price or underlying_price is NaN (due to non-numeric values)
#data = data.dropna(subset=['Strike Price', 'underlying_price'])

# Filter strike prices within the specified range
spot_price = data['underlying_price'].iloc[0]  # Ensure this is numeric now
data = data[(data['Strike Price'] >= spot_price * min_strike_percent) &
            (data['Strike Price'] <= spot_price * max_strike_percent)]

# Calculate implied volatilities
data['Vega_implied_volatility'] = data.apply(
    lambda row: implied_volatility(
        row['mark_price'], row['underlying_price'], row['Strike Price'], 
        row['Time to Expiration'], interest_rate, row['mark_iv'] / 100
    ), axis=1
)

# 3D Surface Plot
strikes = data['Strike Price'].values
times_to_expiration = data['Time to Expiration'].values
implied_vols = data['Vega_implied_volatility'].values

X, Y = np.meshgrid(np.unique(strikes), np.unique(times_to_expiration))
Z = griddata((strikes, times_to_expiration), implied_vols, (X, Y), method='linear')

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn_r')])
fig.update_layout(
    title="Implied Volatility Surface (Vega Approach)",
    scene=dict(
        xaxis_title='Strike Price',
        yaxis_title='Time to Expiry (Years)',
        zaxis_title='Implied Volatility %'
    )
)
st.plotly_chart(fig)
