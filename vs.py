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


#OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#Deribit API 
client_id = 'TsH-x5Hf'
client_secret = 'YR_pRWYuCL91j6Yj9MQpzr8QSO_zO8ZoOrZ2CQjXF2A'
#client_secret = os.getenv("DERIBIT_API_SECRET")



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
    #time.sleep(0.5)
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
coin = st.sidebar.selectbox(
    "Choose a coin:", ["BTC", "ETH", "USDC"],
    help="Due to API restrictions only BTC and ETH options are currently only available" )

st.title(f"Defi Options - {coin}")
st.title("Implied Volatility Surface", help="A volatility surface is a three-dimensional plot of IVs of various options listed on the same underlying. It can be used to visualize the volatility smile/skew and term structure. We use cryptocurrency option data from Deribit to construct volatility surfaces using The Black-Scholes Model.")
st.write("---")

settlement_per = st.sidebar.selectbox(
    "Choose Settlement Period:",
    ['week', 'month'],
    index=0,
    help="Approximate execution times:\n- Month: 30 sec\n- Week: 25 sec"
)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.3f")
strike_range = st.sidebar.slider("Strike Price Range (% of Spot Price)", 0.5, 2.0, (0.50, 2.00))

# Initialize session state to track if the "Run" button has been clicked
if "run_clicked" not in st.session_state:
    st.session_state["run_clicked"] = False

 
run_button = st.sidebar.button("Run")

# Check if the button is clicked
if run_button:
    st.session_state["run_clicked"] = True  
    
    # Your specified code block
    st.subheader(f"Settlement Period: {settlement_per.capitalize()}")

    if settlement_per == "day":
        st.write("EST: 15 sec")
    elif settlement_per == "week":
        st.write("EST: 25 sec")
    else:
        st.write("EST: 30 sec")

    st.write("Fetching data...")
    
 
    data = get_option_data(coin, settlement_per)
    st.session_state["data"] = data   

# Main UI Logic
if not st.session_state["run_clicked"]:
    # Show this message before the "Run" button  
    st.markdown("### Please fill out the parameters and click 'Run'")
else:
    # After "Run" is clicked
    data = st.session_state.get("data")   

    if data is None or data.empty:
        st.write("No data available. Adjust parameters and try again.")
    else:
        # Data successfully fetched 
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

        # GPT prompt  
        prompt = f"""
        You are an experienced quantitative analyst at akuna's crypto trade desk, that focuses on onchain crypto data. Search about the recent crypto news/events, and analyze the following options data, which will be used to generate an implied volatility surface plot for options, given the settlement period of {settlement_per} Based on this analysis, be very short, concise, and provide specific trading strategies that could be effective.
        Consider strategies that take advantage of volatility trends, expiration dates, and strike prices specific to the {coin} options market. Additionally, suggest any hedging or speculative approaches suitable for different market conditions.

        {coin} Options Data with 'Strike Price','Time to Expiration','Vega_implied_volatility':
        {data[['Strike Price', 'Time to Expiration', 'BSM_implied_volatility','Option Type']].to_string(index=False)}

        Only give top 2 strategies, and make sure the who output is max 200 tokens
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            #model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant and quantitative analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        strategies = response['choices'][0]['message']['content'].strip()

        # Display the recommended trading strategies first
        with st.expander("Recommended Trading Strategies"):
            st.markdown(f"### Trading Strategies for {coin.upper()} Options")
            st.write(strategies)

        # Generate the implied volatility surface plot
        fig = go.Figure(data=[go.Surface(
            z=Z_fine, x=X_fine, y=Y_fine, colorscale='RdYlGn_r',
            colorbar=dict(title="I.V. %")
        )])

        fig.update_layout(
            title='Implied Volatility Surface',
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
st.markdown("Created by Ethan Falcao | [LinkedIn](https://www.linkedin.com/in/ethan-falcao/)")
