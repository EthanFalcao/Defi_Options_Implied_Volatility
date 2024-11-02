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


# Set your client_id and client_secret
client_id = 'TsH-x5Hf'
client_secret = 'YR_pRWYuCL91j6Yj9MQpzr8QSO_zO8ZoOrZ2CQjXF2A'

def get_auth_token():
    """Authenticate to Deribit and return an access token."""
    url = "https://test.deribit.com/api/v2/public/auth"
    payload = {
        "jsonrpc": "2.0",
        "id": 9929,
        "method": "public/auth",
        "params": {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('result', {}).get('access_token')
    else:
        print(f"Authentication failed: {response.status_code}, {response.json()}")
        return None

def get_option_name_and_settlement(coin, token):
    """Retrieve option names and settlement periods with authentication."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://deribit.com/api/v2/public/get_instruments?currency={coin}&kind=option"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        name = pd.json_normalize(result['result'])['instrument_name']
        settlement_period = pd.json_normalize(result['result'])['settlement_period']
        return list(name), list(settlement_period)
    else:
        print("Failed to fetch data:", response.status_code)
        return None, None

def extract_details(instrument_name, coin):
    """Extract expiration date, strike price, and option type (call or put) from instrument name."""
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-([CP])", instrument_name)
    if match:
        expiration_date = match.group(1)
        strike_price = match.group(2)
        option_type = 'Call' if match.group(3) == 'C' else 'Put'
        return expiration_date, strike_price, option_type
    return None, None, None

def fetch_option_data(option_name, token):
    """Fetch the option data for a given option name with authentication."""
    time.sleep(0.3)
    headers = {"Authorization": f"Bearer {token}"}
    url = f'https://deribit.com/api/v2/public/get_order_book?instrument_name={option_name}'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        df = pd.json_normalize(result['result'])
        selected_columns = ["instrument_name", "mark_price", "underlying_price", "mark_iv", "greeks.vega"]
        return df[selected_columns]
    else:
        print(f"Failed to fetch option data for {option_name}: {response.status_code}")
        return None

def get_option_data(coin, settlement_per):
    """Main function to get and process options data."""
    token = get_auth_token()
    if not token:
        print("Token retrieval failed.")
        return None
    
    coin_name, settlement_period = get_option_name_and_settlement(coin, token)
    if settlement_per not in settlement_period:
        print(f"No options available with settlement period '{settlement_per}'.")
        return None
    
    coin_name_filtered = [coin_name[i] for i in range(len(coin_name)) if settlement_period[i] == settlement_per]
    pbar = tqdm(total=len(coin_name_filtered))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_option = {executor.submit(fetch_option_data, name, token): name for name in coin_name_filtered}
        coin_df = []
        for future in concurrent.futures.as_completed(future_to_option):
            try:
                data = future.result()
                if data is not None:
                    data['settlement_period'] = settlement_per
                    coin_df.append(data)
            except Exception as exc:
                print(f'Error fetching data: {exc}')
            pbar.update(1)
    
    if coin_df:
        coin_df = pd.concat(coin_df, ignore_index=True)
    else:
        print("No data fetched.")
        return None
    
    coin_df['Expiration Date'], coin_df['Strike Price'], coin_df['Option Type'] = zip(*coin_df['instrument_name'].apply(lambda x: extract_details(x, coin)))
    today = datetime.today()
    coin_df['Time to Expiration'] = coin_df['Expiration Date'].apply(lambda x: (datetime.strptime(x, '%d%b%y') - today).days / 365 if x else None)
    final_columns = ["instrument_name", "Option Type", "Expiration Date", "Strike Price", 'Time to Expiration', 'mark_price', 'underlying_price', 'mark_iv', 'greeks.vega', 'settlement_period']
    coin_df = coin_df[final_columns]
    pbar.close()
    
    return coin_df


# Define Streamlit UI components
st.sidebar.header("Parameters")
coin = st.sidebar.selectbox("Choose a coin:", ['BTC', 'ETH'])

st.title(f"Defi Options - {coin}")
st.title("Implied Volatility Surface")


settlement_per = st.sidebar.selectbox(
    "Choose Settlement Period:",
    ['day','week', 'month', ],
    index=1, 
    help="Approximate execution times:\n- Month: 1.5 min\n- Week: 45 sec\n- Day: 15 sec"
)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.3f")
strike_range = st.sidebar.slider("Strike Price Range (% of Spot Price)", 0.5, 2.0, (0.50, 1.70))
st.subheader(f"Settlement Period: {settlement_per.capitalize()}")

# Black-Scholes Model and Vega Functions
   
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    # Ensure T and sigma are non-negative
    if T <= 0 or sigma <= 0:
        return 0  # Or return None, depending on how you want to handle it
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    
def vega(S, K, T, r, sigma):
    # Ensure T and sigma are non-negative
    if T <= 0 or sigma <= 0:
        return 0  # Or return None
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

# Data Fetching and Processing
if settlement_per == "day":
    st.write("ETA: 15 sec")
elif settlement_per == "week":
    st.write("ETA: 45 sec")
else:
    st.write("ETA: 1.5 min")
    
    
st.write("Fetching data...")
data = get_option_data(coin, settlement_per)
if data is None or data.empty:
    st.write("No data available.")
else:
    st.write("Data fetched successfully.")
    


# Ensure Strike Price is numeric
data['Strike Price'] = pd.to_numeric(data['Strike Price'], errors='coerce').astype('float64')
btc_data = data.dropna()

# Apply strike price filter based on user input
min_strike, max_strike = strike_range
btc_data = btc_data[(btc_data['Strike Price'] >= btc_data['underlying_price'] * min_strike) & 
                    (btc_data['Strike Price'] <= btc_data['underlying_price'] * max_strike)]

# Calculate implied volatility for each option
results = []
for index, row in btc_data.iterrows():
    S, K, T = row['underlying_price'], row['Strike Price'], row['Time to Expiration']
    market_price, initial_vol = row['mark_price'], row['mark_iv'] / 100
    iv = implied_volatility(market_price, S, K, T, interest_rate, initial_vol, option_type="call")
    results.append(iv)
btc_data['BSM_implied_volatility'] = results  # Store results in a new column

# Define the columns needed from your data
strikes = btc_data['Strike Price'].values
times_to_expiration = btc_data['Time to Expiration'].values
implied_vols = btc_data['BSM_implied_volatility'].values

# Define a finer grid resolution for strikes and times_to_expiration
num_points = 100  # Adjust for higher or lower resolution; higher values = more detail
fine_strikes = np.linspace(strikes.min(), strikes.max(), num_points)
fine_times_to_expiration = np.linspace(times_to_expiration.min(), times_to_expiration.max(), num_points)

# Create a meshgrid for the finer grid
X_fine, Y_fine = np.meshgrid(fine_strikes, fine_times_to_expiration)

# Interpolate the implied volatilities to fill the finer grid
Z_fine = griddata((strikes, times_to_expiration), implied_vols, (X_fine, Y_fine), method='linear')

# Create the interactive 3D surface plot with color scale title
fig = go.Figure(data=[go.Surface(
    z=Z_fine, x=X_fine, y=Y_fine, colorscale='RdYlGn_r',
    colorbar=dict(title="I.V. %")  # Title for the color scale
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
        aspectmode="cube"  # Ensures equal aspect ratio for x, y, and z
    )
)

# Display the plot in Streamlit
st.plotly_chart(fig)


st.write("---")
st.markdown("Created by Ethan Falcao | [LinkedIn](https://www.linkedin.com/in/ethan-falcao//)")
