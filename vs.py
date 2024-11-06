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
import openai
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set your Deribit API client_id and client_secret
client_id = 'TsH-x5Hf'
client_secret = 'YR_pRWYuCL91j6Yj9MQpzr8QSO_zO8ZoOrZ2CQjXF2A'

# Authentication and data fetching functions
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
    time.sleep(0.5)
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

# Black-Scholes Model and Vega Functions
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0  
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
def vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0  
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

# Streamlit Sidebar and Parameters
st.sidebar.header("Parameters")
coin = st.sidebar.selectbox("Choose a coin:", ['BTC', 'ETH'])

st.title(f"Defi Options - {coin}")
st.title("Implied Volatility Surface")

settlement_per = st.sidebar.selectbox(
    "Choose Settlement Period:",
    ['day', 'week', 'month'],
    index=1,
    help="Approximate execution times:\n- Month: 30 sec\n- Week: 25 sec\n- Day: 15 sec "
)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.3f")
strike_range = st.sidebar.slider("Strike Price Range (% of Spot Price)", 0.5, 2.0, (0.50, 2.00))
st.subheader(f"Settlement Period: {settlement_per.capitalize()}")




if settlement_per == "day":
    st.write("EST: 15 sec")
elif settlement_per == "week":
    st.write("EST: 25 sec")
else:
    st.write("EST: 30 sec")

st.write("Fetching data...")

data = get_option_data(coin, settlement_per)
    
    
    

if data is None or data.empty:
    st.write("No data available.")
else:
    st.write("Data fetched successfully.")
    
    data['Strike Price'] = pd.to_numeric(data['Strike Price'], errors='coerce').astype('float64')
    data = data.dropna()

    min_strike, max_strike = strike_range
    data = data[(data['Strike Price'] >= data['underlying_price'] * min_strike) & 
                        (data['Strike Price'] <= data['underlying_price'] * max_strike)]

    results = []
    for index, row in data.iterrows():
        S, K, T = row['underlying_price'], row['Strike Price'], row['Time to Expiration']
        market_price, initial_vol = row['mark_price'], row['mark_iv'] / 100
        iv = implied_volatility(market_price, S, K, T, interest_rate, initial_vol, option_type="call")
        results.append(iv)
    data['BSM_implied_volatility'] = results

    strikes = data['Strike Price'].values
    times_to_expiration = data['Time to Expiration'].values
    implied_vols = data['BSM_implied_volatility'].values

    num_points = 100  
    fine_strikes = np.linspace(strikes.min(), strikes.max(), num_points)
    fine_times_to_expiration = np.linspace(times_to_expiration.min(), times_to_expiration.max(), num_points)
    X_fine, Y_fine = np.meshgrid(fine_strikes, fine_times_to_expiration)
    Z_fine = griddata((strikes, times_to_expiration), implied_vols, (X_fine, Y_fine), method='linear')

    fig = go.Figure(data=[go.Surface(
        z=Z_fine, x=X_fine, y=Y_fine, colorscale='RdYlGn_r',
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
    
    
    
    
    # Define a prompt to request insights based on the data summaries
    prompt = f"""
    You are a quantitative analyst. Please analyze the following options data, which will be used to generate an implied volatility surface plot for options. Based on this analysis, be very short, concise, and provide specific trading strategies that could be effective.
    Consider strategies that take advantage of volatility trends, expiration dates, and strike prices specific to the {coin} options market. Additionally, suggest any hedging or speculative approaches suitable for different market conditions.

    {coin} Options Data with 'Strike Price','Time to Expiration','Vega_implied_volatility':
    {data[['Strike Price', 'Time to Expiration', 'BSM_implied_volatility']].to_string(index=False)}

    Only give top 3 strategies.
    """


    
    # Generate the completion using the updated API format with ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Ensure the model name is correct
        messages=[
            {"role": "system", "content": "You are a helpful assistant and quantitative analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,  # Adjust token limit as needed
        temperature=0.7
    )

    # Retrieve and print the response
    strategies = response['choices'][0]['message']['content'].strip()
    

        
        
    with st.expander("Recommended Trading Strategies"):
        st.markdown(f"### Top 3 Trading Strategies for {coin.upper()} Options")
        st.write(strategies)


st.write("---")
st.markdown("Created by Ethan Falcao | [LinkedIn](https://www.linkedin.com/in/ethan-falcao/)")
