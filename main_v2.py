import streamlit as st
import pandas as pd
import json
import requests
import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go
import concurrent.futures
import re
from tqdm import tqdm

# Function to fetch option names and settlement
def get_option_name_and_settlement(coin):
    r = requests.get(f"https://test.deribit.com/api/v2/public/get_instruments?currency={coin}&kind=option")
    result = json.loads(r.text)
    name = pd.json_normalize(result['result'])['instrument_name'].tolist()
    settlement_period = pd.json_normalize(result['result'])['settlement_period'].tolist()
    return name, settlement_period

# Function to fetch option data
def fetch_option_data(option_name):
    r = requests.get(f'https://test.deribit.com/api/v2/public/get_order_book?instrument_name={option_name}')
    result = json.loads(r.text)
    return pd.json_normalize(result['result'])

# Extract details
def extract_details(instrument_name, coin):
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-([CP])", instrument_name)
    if match:
        expiration_date, strike_price, option_type = match.group(1), match.group(2), 'Call' if match.group(3) == 'C' else 'Put'
        return expiration_date, strike_price, option_type
    return None, None, None

# Fetch option data for selected coin and settlement period
def get_option_data(coin, settlement_per):
    coin_name, settlement_period = get_option_name_and_settlement(coin)
    coin_name_filtered = [coin_name[i] for i in range(len(coin_name)) if settlement_period[i] == settlement_per]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        coin_df = [future.result() for future in 
                   [executor.submit(fetch_option_data, name) for name in coin_name_filtered] if future.result() is not None]
    if coin_df:
        coin_df = pd.concat(coin_df)
        coin_df['Expiration Date'], coin_df['Strike Price'], coin_df['Option Type'] = zip(*coin_df['instrument_name'].apply(lambda x: extract_details(x, coin)))
        coin_df['Time to Expiration'] = coin_df['Expiration Date'].apply(lambda x: (datetime.strptime(x, '%d%b%y') - datetime.today()).days / 365 if x else None)
    return coin_df

# Black-Scholes functions
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(market_price, S, K, T, r, initial_vol, option_type="call", tolerance=1e-5, max_iterations=100):
    sigma = initial_vol
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega_value = vega(S, K, T, r, sigma)
        if vega_value < 1e-5:
            break
        sigma += (market_price - price) / vega_value
        if abs(market_price - price) < tolerance:
            return sigma
    return None

# Streamlit app
st.title("Crypto Options Implied Volatility Surface")
st.sidebar.header("User Inputs")

# User inputs
coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH"])
settlement_period = st.sidebar.selectbox("Settlement Period", ["day", "week", "month"])
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0) / 100

# Fetch and process data
data = get_option_data(coin, settlement_period)
if data.empty:
    st.write("No options data available for the selected coin and settlement period.")
else:
    # Select relevant columns and filter data within strike price range
    data = data[["instrument_name", "Option Type", "mark_price", "underlying_price", "mark_iv", "greeks.vega", "Expiration Date", "Strike Price", "Time to Expiration"]]
    min_strike, max_strike = 0.73 * data['underlying_price'].mean(), 1.20 * data['underlying_price'].mean()
    filtered_data = data[(data['Strike Price'] >= min_strike) & (data['Strike Price'] <= max_strike)]

    # Calculate Vega-implied volatility
    results = []
    for _, row in filtered_data.iterrows():
        iv = implied_volatility(row['mark_price'], row['underlying_price'], row['Strike Price'], row['Time to Expiration'], interest_rate, row['mark_iv'] / 100, row['Option Type'].lower())
        results.append(iv)
    filtered_data['Vega_implied_volatility'] = results

    # Plot
    strikes, times_to_expiration, implied_vols = filtered_data['Strike Price'], filtered_data['Time to Expiration'], filtered_data['Vega_implied_volatility']
    X, Y = np.meshgrid(np.unique(strikes), np.unique(times_to_expiration))
    Z = griddata((strikes, times_to_expiration), implied_vols, (X, Y), method='linear')

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdYlGn_r', colorbar=dict(title="Implied Volatility %"))])
    fig.update_layout(title='Implied Volatility Surface (Vega Approach)', width=700, height=700,
                      scene=dict(xaxis_title='Strike Price', yaxis_title='Time to Expiry (Years)', zaxis_title='Implied Volatility %'))

    st.plotly_chart(fig)
