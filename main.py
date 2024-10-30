import pandas as pd
import json
import requests
import re
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Functions
def get_option_name_and_settlement(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: 2 lists:
                        1. list of traded options for the selected coin;
                        2. list of settlement period for the selected coin.
    """
    r = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option")
    result = json.loads(r.text)
    # Get option name
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)

    # Get option settlement period
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)

    return name, settlement_period

import time  # Add import for time

def fetch_option_data(option_name):
    """Fetch the option data for a given option name with a small delay to avoid rate limiting."""
    time.sleep(0.1)  # Add a short delay to avoid hitting rate limits
    r = requests.get(f'https://test.deribit.com/api/v2/public/get_order_book?instrument_name={option_name}')
    result = json.loads(r.text)
    return pd.json_normalize(result['result'])


def extract_details(instrument_name, coin):
    """
    Extract expiration date, strike price, and option type (call or put) from instrument name.
    Adjusts to include selected coin (e.g., 'BTC' or 'ETH').
    """
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-([CP])", instrument_name)
    if match:
        expiration_date = match.group(1)
        strike_price = match.group(2)
        option_type = 'Call' if match.group(3) == 'C' else 'Put'
        return expiration_date, strike_price, option_type
    return None, None, None


def get_option_data(coin, settlement_per):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: pandas DataFrame with all option data for a given coin, filtered for options with specified settlement.
    """
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
                if not data.empty:  # Ensure non-empty data
                    coin_df.append(data)
            except Exception as exc:
                print(f'Error fetching data: {exc}')
            pbar.update(1)

    # Finalize DataFrame
    if len(coin_df) > 0:
        coin_df = pd.concat([df for df in coin_df if not df.empty])

    # Remove unnecessary columns
    columns = ['state', 'estimated_delivery_price']
    if not coin_df.empty:
        coin_df.drop(columns, inplace=True, axis=1)

    coin_df['Expiration Date'], coin_df['Strike Price'], coin_df['Option Type'] = zip(*coin_df['instrument_name'].apply(lambda x: extract_details(x, coin)))
    
    today = datetime.today()
    coin_df['Time to Expiration'] = coin_df['Expiration Date'].apply(lambda x: (datetime.strptime(x, '%d%b%y') - today).days / 365 if x else None)

    coin_df.to_csv('data/data.csv', index=False)
 
    pbar.close()

    return coin_df

data = get_option_data('BTC','month')
data = data[["instrument_name", "Option Type", 'mark_price', 'underlying_price', 'mark_iv', 'greeks.vega', 'Expiration Date', 'Strike Price', 'Time to Expiration']]

data['Strike Price'] = pd.to_numeric(data['Strike Price'], errors='coerce').astype('float64')
btc_data = data.dropna()

# Vega and Black-Scholes calculations
interest_rate = 0.05

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return None  # Handle cases where sigma or T is zero or negative

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return price

def vega(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0  # Handle cases where sigma or T is zero or negative

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(market_price, S, K, T, r, initial_vol, option_type="call", tolerance=1e-5, max_iterations=100):
    sigma = initial_vol
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        if price is None:
            return None
        
        vega_value = vega(S, K, T, r, sigma)
        
        if vega_value < 1e-5:  # Check if Vega is too small
            break
        
        price_difference = market_price - price
        sigma += price_difference / vega_value
        
        if abs(price_difference) < tolerance:
            return sigma
    
    return None

# Apply implied volatility calculation
results = []
for index, row in btc_data.iterrows():
    S = row['underlying_price']
    K = row['Strike Price']
    T = row['Time to Expiration']
    r = interest_rate
    market_price = row['mark_price']
    initial_vol = row['mark_iv'] / 100  # Assuming mark_iv is in percentage
    option_type = row['Option Type'].lower()  # Use "call" or "put" based on the data
    
    # Calculate implied volatility
    iv = implied_volatility(market_price, S, K, T, r, initial_vol, option_type=option_type)
    results.append(iv)

# Add the results to the DataFrame
btc_data['Vega_implied_volatility'] = results

# 3D Plot of Implied Volatility
strikes = btc_data['Strike Price'].values
times_to_expiration = btc_data['Time to Expiration'].values
implied_vols = btc_data['Vega_implied_volatility'].values

# Create meshgrid for strikes and times_to_expiration
X, Y = np.meshgrid(np.unique(strikes), np.unique(times_to_expiration))

# Interpolate the implied volatilities to fill the grid
Z = griddata((strikes, times_to_expiration), implied_vols, (X, Y), method='nearest')

# Create the interactive 3D surface plot with color scale title
fig = go.Figure(data=[go.Surface(
    z=Z, x=X, y=Y, colorscale='RdYlGn_r',  # Green-to-red color scale
    colorbar=dict(title="Implied Volatility %")
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(title='Strike Price', range=[strikes.min(), strikes.max()]),
        yaxis=dict(title='Time to Expiry (Years)', range=[times_to_expiration.min(), times_to_expiration.max()]),
        zaxis=dict(title='Implied Volatility %', range=[min(implied_vols), max(implied_vols)])
    )
)

fig.show()
