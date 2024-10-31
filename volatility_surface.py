# app.py

import streamlit as st
import pandas as pd
import json
import requests
import re
from datetime import datetime
import time
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.interpolate import griddata
import concurrent.futures
from tqdm import tqdm

# Define functions
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
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
        price_difference = market_price - price
        sigma += price_difference / vega_value
        if abs(price_difference) < tolerance:
            return sigma
    return None

# API key
API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
# Define headers for authenticated requests
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
# Functions
def get_option_name_and_settlement(coin):
    r = requests.get(f"https://test.deribit.com/api/v2/public/get_instruments?currency={coin}&kind=option", headers=headers)
    result = json.loads(r.text)
    # Get option name
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)
    # Get option settlement period
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)

    return name, settlement_period 

def fetch_option_data(option_name):
    """Fetch the option data for a given option name with a small delay to avoid rate limiting, and select only specific columns."""
    time.sleep(0.1)  # Add a short delay to avoid hitting rate limits
    r = requests.get(f'https://test.deribit.com/api/v2/public/get_order_book?instrument_name={option_name}')
    result = json.loads(r.text)
    
    # Normalize the JSON data and filter for required columns
    df = pd.json_normalize(result['result'])
    selected_columns = ["instrument_name", "mark_price", "underlying_price", "mark_iv", "greeks.vega"]
    return df[selected_columns]

def extract_details(instrument_name, coin):
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-([CP])", instrument_name)
    if match:
        expiration_date = match.group(1)
        strike_price = match.group(2)
        option_type = 'Call' if match.group(3) == 'C' else 'Put'
        return expiration_date, strike_price, option_type
    return None, None, None

def get_option_data(coin, settlement_per):
    # Get option name and settlement
    coin_name, settlement_period = get_option_name_and_settlement(coin)
    # Filter options that have the specified settlement period
    coin_name_filtered = [coin_name[i] for i in range(len(coin_name)) if settlement_period[i] == settlement_per]
    # Initialize progress bar
    pbar = tqdm(total=len(coin_name_filtered))

    # Fetch data concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_option = {executor.submit(fetch_option_data, name): name for name in coin_name_filtered}
        coin_df = []
        for future in concurrent.futures.as_completed(future_to_option):
            try:
                data = future.result()
                data['settlement_period'] = settlement_per
                coin_df.append(data)
            except Exception as exc:
                print(f'Error fetching data: {exc}')
            pbar.update(1)

    # Finalize DataFrame
    if len(coin_df) > 0:
        coin_df = pd.concat(coin_df)
    # Extract expiration date, strike price, and option type
    coin_df['Expiration Date'], coin_df['Strike Price'], coin_df['Option Type'] = zip(*coin_df['instrument_name'].apply(lambda x: extract_details(x, coin)))
    # Calculate time to expiration
    today = datetime.today()
    coin_df['Time to Expiration'] = coin_df['Expiration Date'].apply(lambda x: (datetime.strptime(x, '%d%b%y') - today).days / 365 if x else None)
    # Select the final columns
    final_columns = ["instrument_name", "Option Type", 'mark_price', 'underlying_price', 'mark_iv', 'greeks.vega', 'Expiration Date', 'Strike Price', 'Time to Expiration']
    coin_df = coin_df[final_columns]
    coin_df.to_csv('data/data.csv', index=False)
    pbar.close()
    return coin_df

st.sidebar.header("Parameters")
coin = st.sidebar.selectbox("Choose a coin:", ['BTC', 'ETH'])

# Streamlit Interface
st.title(f"Defi Options - {coin}")
st.title("Implied Volatility Surface")


# Sidebar inputs

# Add a note about expected data retrieval times
#st.sidebar.markdown("**Settlement Period:**")
settlement_per = st.sidebar.selectbox(
    "Choose Settlement Period:",
    ['day','week','month'],
    help="Approximate execution times:\n- Month: 1.5 min\n- Week: 45 sec\n- Day: 15 sec"
)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.015, step=0.001,format="%.3f")




strike_range = st.sidebar.slider("Strike Price Range (% of Spot Price)", 0.5, 2.0, (0.73, 1.20))

# Display chosen settings under the title
st.subheader(f" Settlement Period: {settlement_per.capitalize()}")

st.write("Fetching data...")

# Data Fetching and Processing
data = get_option_data(coin, settlement_per)
if data.empty:
    st.write("No data available.")
else:
    st.write("Data fetched successfully.")
    data = data[["instrument_name", "Option Type", 'mark_price', 'underlying_price', 'mark_iv', 'greeks.vega', 'Expiration Date', 'Strike Price', 'Time to Expiration']]
    data['Strike Price'] = pd.to_numeric(data['Strike Price'], errors='coerce').astype('float64')
    btc_data = data.dropna()

    # Apply strike price filter based on user input
    min_strike, max_strike = strike_range
    btc_data = btc_data[(btc_data['Strike Price'] >= btc_data['underlying_price'] * min_strike) &
                        (btc_data['Strike Price'] <= btc_data['underlying_price'] * max_strike)]

    results = []
    for index, row in btc_data.iterrows():
        S = row['underlying_price']
        K = row['Strike Price']
        T = row['Time to Expiration']
        r = interest_rate
        market_price = row['mark_price']
        initial_vol = row['mark_iv'] / 100
        iv = implied_volatility(market_price, S, K, T, r, initial_vol, option_type="call")
        results.append(iv)
    btc_data['Vega_implied_volatility'] = results

    # Surface Plot
    strikes = btc_data['Strike Price'].values
    times_to_expiration = btc_data['Time to Expiration'].values
    implied_vols = btc_data['Vega_implied_volatility'].values
    X, Y = np.meshgrid(np.unique(strikes), np.unique(times_to_expiration))
    Z = griddata((strikes, times_to_expiration), implied_vols, (X, Y), method='linear')

    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y, colorscale='RdYlGn_r',
        colorbar=dict(title="I.V. %")
    )])

    fig.update_layout(
        title='Implied Volatility Surface (Vega-Based Iterative Approach)',
        autosize=False,
        width=700,
        height=700,
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Expiry (Years)',
            zaxis_title='Implied Volatility %',
            xaxis=dict(type="log"),
            aspectmode="cube"
        )
    )
    st.plotly_chart(fig)
    
    
st.write("---")
st.markdown(
    "Created by Ethan Falcao  |   [LinkedIn](https://www.linkedin.com/in/ethan-falcao//)"
)
