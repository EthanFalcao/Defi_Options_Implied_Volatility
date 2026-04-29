import streamlit as st
import pandas as pd
import requests
import re
from datetime import datetime
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.interpolate import griddata
import concurrent.futures
from openai import OpenAI
import os


# ─────────────────────────────────────────────
# API SETUP
# ─────────────────────────────────────────────

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Better: use environment variables for Deribit credentials.
# In PowerShell:
# $env:DERIBIT_CLIENT_ID="your_client_id"
# $env:DERIBIT_API_SECRET="your_client_secret"

client_id = os.getenv("DERIBIT_CLIENT_ID", "TsH-x5Hf")
client_secret = os.getenv("DERIBIT_API_SECRET", "YR_pRWYuCL91j6Yj9MQpzr8QSO_zO8ZoOrZ2CQjXF2A")


# ─────────────────────────────────────────────
# DERIBIT DATA FUNCTIONS
# ─────────────────────────────────────────────

def get_auth_token():
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

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)

        if response.status_code == 200:
            return response.json().get("result", {}).get("access_token")

        st.error(f"Authentication failed: {response.status_code}")
        st.write(response.text)
        return None

    except Exception as e:
        st.error(f"Authentication request failed: {e}")
        return None


def get_option_name_and_settlement(coin, token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://deribit.com/api/v2/public/get_instruments?currency={coin}&kind=option"

    try:
        response = requests.get(url, headers=headers, timeout=20)

        if response.status_code == 200:
            result = response.json()
            instruments = pd.json_normalize(result["result"])

            if "instrument_name" not in instruments.columns or "settlement_period" not in instruments.columns:
                st.error("Missing expected instrument columns from Deribit response.")
                return None, None

            name = instruments["instrument_name"]
            settlement_period = instruments["settlement_period"]

            return list(name), list(settlement_period)

        st.error(f"Failed to fetch instruments: {response.status_code}")
        st.write(response.text)
        return None, None

    except Exception as e:
        st.error(f"Instrument request failed: {e}")
        return None, None


def extract_details(instrument_name, coin):
    match = re.match(fr"{coin}-(\d+[A-Z]{{3}}\d+)-(\d+)-([CP])", instrument_name)

    if match:
        expiration_date = match.group(1)
        strike_price = match.group(2)
        option_type = "Call" if match.group(3) == "C" else "Put"
        return expiration_date, strike_price, option_type

    return None, None, None


def fetch_option_data(option_name, token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://deribit.com/api/v2/public/get_order_book?instrument_name={option_name}"

    try:
        response = requests.get(url, headers=headers, timeout=20)

        if response.status_code == 200:
            result = response.json()
            df = pd.json_normalize(result["result"])

            selected_columns = [
                "instrument_name",
                "mark_price",
                "underlying_price",
                "mark_iv",
                "greeks.vega"
            ]

            missing_columns = [col for col in selected_columns if col not in df.columns]

            if missing_columns:
                return None

            return df[selected_columns]

        print(f"Failed to fetch option data for {option_name}: {response.status_code}")
        return None

    except Exception as e:
        print(f"Request failed for {option_name}: {e}")
        return None


def get_option_data(coin, settlement_per):
    token = get_auth_token()

    if not token:
        st.error("Token retrieval failed.")
        return None

    coin_name, settlement_period = get_option_name_and_settlement(coin, token)

    if coin_name is None or settlement_period is None:
        return None

    if settlement_per not in settlement_period:
        st.error(f"No options available with settlement period '{settlement_per}'.")
        return None

    coin_name_filtered = [
        coin_name[i]
        for i in range(len(coin_name))
        if settlement_period[i] == settlement_per
    ]

    if not coin_name_filtered:
        st.error("No matching option instruments found.")
        return None

    coin_df = []

    progress_text = st.empty()
    progress_bar = st.progress(0)

    total = len(coin_name_filtered)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_option = {
            executor.submit(fetch_option_data, name, token): name
            for name in coin_name_filtered
        }

        completed = 0

        for future in concurrent.futures.as_completed(future_to_option):
            try:
                data = future.result()

                if data is not None and not data.empty:
                    data["settlement_period"] = settlement_per
                    coin_df.append(data)

            except Exception as exc:
                print(f"Error fetching data: {exc}")

            completed += 1
            progress_bar.progress(completed / total)
            progress_text.write(f"Fetched {completed} of {total} instruments...")

    progress_text.empty()
    progress_bar.empty()

    if not coin_df:
        st.error("No data fetched.")
        return None

    coin_df = pd.concat(coin_df, ignore_index=True)

    details = coin_df["instrument_name"].apply(lambda x: extract_details(x, coin))
    coin_df["Expiration Date"], coin_df["Strike Price"], coin_df["Option Type"] = zip(*details)

    today = datetime.today()

    coin_df["Time to Expiration"] = coin_df["Expiration Date"].apply(
        lambda x: (datetime.strptime(x, "%d%b%y") - today).days / 365 if x else None
    )

    final_columns = [
        "instrument_name",
        "Option Type",
        "Expiration Date",
        "Strike Price",
        "Time to Expiration",
        "mark_price",
        "underlying_price",
        "mark_iv",
        "greeks.vega",
        "settlement_period"
    ]

    coin_df = coin_df[final_columns]

    return coin_df


# ─────────────────────────────────────────────
# VIX-STYLE FORMULA
# ─────────────────────────────────────────────

def calculate_vix(df, r):
    df = df.copy()

    df["Strike Price"] = pd.to_numeric(df["Strike Price"], errors="coerce")

    df = df.dropna(
        subset=[
            "Strike Price",
            "mark_price",
            "Time to Expiration",
            "underlying_price"
        ]
    )

    if df.empty:
        return None, None

    T = df["Time to Expiration"].iloc[0]

    if T <= 0:
        return None, None

    calls = (
        df[df["Option Type"] == "Call"][["Strike Price", "mark_price"]]
        .rename(columns={"mark_price": "call_price"})
    )

    puts = (
        df[df["Option Type"] == "Put"][["Strike Price", "mark_price"]]
        .rename(columns={"mark_price": "put_price"})
    )

    pairs = pd.merge(calls, puts, on="Strike Price").dropna()

    if pairs.empty:
        return None, None

    pairs["diff"] = abs(pairs["call_price"] - pairs["put_price"])

    atm_row = pairs.loc[pairs["diff"].idxmin()]

    K_atm = atm_row["Strike Price"]
    C_atm = atm_row["call_price"]
    P_atm = atm_row["put_price"]

    F = K_atm + np.exp(r * T) * (C_atm - P_atm)

    strikes_sorted = sorted(df["Strike Price"].unique())
    strikes_below = [k for k in strikes_sorted if k <= F]

    if not strikes_below:
        return None, None

    K0 = max(strikes_below)

    otm_puts = df[
        (df["Option Type"] == "Put") &
        (df["Strike Price"] < K0)
    ]

    otm_calls = df[
        (df["Option Type"] == "Call") &
        (df["Strike Price"] > K0)
    ]

    at_k0 = df[df["Strike Price"] == K0]

    if not at_k0.empty:
        q_k0 = at_k0["mark_price"].mean()
        k0_df = pd.DataFrame(
            {
                "Strike Price": [K0],
                "mark_price": [q_k0]
            }
        )
    else:
        k0_df = pd.DataFrame(columns=["Strike Price", "mark_price"])

    otm = pd.concat(
        [
            otm_puts[["Strike Price", "mark_price"]],
            k0_df,
            otm_calls[["Strike Price", "mark_price"]]
        ]
    )

    otm = (
        otm.drop_duplicates("Strike Price")
        .sort_values("Strike Price")
        .reset_index(drop=True)
    )

    if len(otm) < 2:
        return None, None

    K_vals = otm["Strike Price"].values
    delta_K = np.zeros(len(K_vals))

    for i in range(len(K_vals)):
        if i == 0:
            delta_K[i] = K_vals[1] - K_vals[0]
        elif i == len(K_vals) - 1:
            delta_K[i] = K_vals[-1] - K_vals[-2]
        else:
            delta_K[i] = (K_vals[i + 1] - K_vals[i - 1]) / 2.0

    Q = otm["mark_price"].values
    K = K_vals
    eRT = np.exp(r * T)

    term1 = (2 / T) * np.sum((delta_K / K**2) * eRT * Q)
    term2 = (1 / T) * ((F / K0) - 1) ** 2

    variance = term1 - term2

    if variance <= 0:
        return None, None

    sigma = np.sqrt(variance)
    vix = sigma * 100

    return sigma, vix


def calculate_crypto_vix(df, r):
    df = df.copy()
    df = df.dropna(subset=["Expiration Date", "Time to Expiration"])

    if df.empty:
        return None

    expirations = sorted(
        df["Expiration Date"].unique(),
        key=lambda x: datetime.strptime(x, "%d%b%y")
    )

    results = []

    for exp in expirations:
        slice_df = df[df["Expiration Date"] == exp]

        if slice_df.empty:
            continue

        T = slice_df["Time to Expiration"].iloc[0]

        if T <= 0:
            continue

        sigma, vix = calculate_vix(slice_df, r)

        if vix is not None:
            results.append(
                {
                    "expiration": exp,
                    "T": T,
                    "sigma": sigma,
                    "vix": vix
                }
            )

    if len(results) == 0:
        return None

    if len(results) == 1:
        return results[0]["vix"]

    r1, r2 = results[0], results[1]

    T1 = r1["T"]
    T2 = r2["T"]

    sigma1_sq = r1["sigma"] ** 2
    sigma2_sq = r2["sigma"] ** 2

    T30 = 30 / 365

    if T2 == T1:
        return r1["vix"]

    crypto_vix_sq = (
        T1 * sigma1_sq * (T2 - T30) / (T2 - T1) +
        T2 * sigma2_sq * (T30 - T1) / (T2 - T1)
    ) * (365 / 30)

    if crypto_vix_sq <= 0:
        return None

    return np.sqrt(crypto_vix_sq) * 100


# ─────────────────────────────────────────────
# BLACK-SCHOLES FUNCTIONS
# ─────────────────────────────────────────────

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0

    d1 = (
        np.log(S / K) +
        (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0

    d1 = (
        np.log(S / K) +
        (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    initial_vol,
    option_type="call",
    tolerance=1e-5,
    max_iterations=100
):
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    sigma = initial_vol

    if sigma is None or sigma <= 0 or np.isnan(sigma):
        sigma = 0.5

    for _ in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega_value = vega(S, K, T, r, sigma)

        if vega_value < 1e-5:
            break

        price_difference = market_price - price
        sigma += price_difference / vega_value

        if sigma <= 0 or np.isnan(sigma) or np.isinf(sigma):
            return None

        if abs(price_difference) < tolerance:
            return sigma

    return sigma if sigma > 0 else None


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.sidebar.header("Parameters")

coin = st.sidebar.selectbox(
    "Choose a coin:",
    ["BTC", "ETH"],
    help="Due to API restrictions only BTC and ETH options are currently available."
)

st.title(f"Defi Options - {coin}")

st.title(
    "Implied Volatility Surface",
    help=(
        "A volatility surface is a three-dimensional plot of implied volatilities "
        "for options listed on the same underlying."
    )
)

st.write("---")

settlement_per = st.sidebar.selectbox(
    "Choose Settlement Period:",
    ["week", "month"],
    index=0,
    help="Approximate execution times:\n- Month: 30 sec\n- Week: 25 sec"
)

interest_rate = st.sidebar.number_input(
    "Interest Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.015,
    step=0.001,
    format="%.3f"
)

strike_range = st.sidebar.slider(
    "Strike Price Range (% of Spot Price)",
    0.5,
    2.0,
    (0.50, 2.00)
)

if "run_clicked" not in st.session_state:
    st.session_state["run_clicked"] = False

run_button = st.sidebar.button("Run")

if run_button:
    st.session_state["run_clicked"] = True

    st.subheader(f"Settlement Period: {settlement_per.capitalize()}")

    if settlement_per == "week":
        st.write("EST: 25 sec")
    elif settlement_per == "month":
        st.write("EST: 30 sec")

    st.write("Fetching data...")

    data = get_option_data(coin, settlement_per)

    st.session_state["data"] = data


if not st.session_state["run_clicked"]:
    st.markdown("### Please fill out the parameters and click 'Run'")

else:
    data = st.session_state.get("data")

    if data is None or data.empty:
        st.write("No data available. Adjust parameters and try again.")

    else:
        st.write("Data fetched successfully.")

        data = data.copy()

        data["Strike Price"] = pd.to_numeric(
            data["Strike Price"],
            errors="coerce"
        )

        data["mark_price"] = pd.to_numeric(
            data["mark_price"],
            errors="coerce"
        )

        data["underlying_price"] = pd.to_numeric(
            data["underlying_price"],
            errors="coerce"
        )

        data["mark_iv"] = pd.to_numeric(
            data["mark_iv"],
            errors="coerce"
        )

        data["Time to Expiration"] = pd.to_numeric(
            data["Time to Expiration"],
            errors="coerce"
        )

        data = data.dropna(
            subset=[
                "Strike Price",
                "mark_price",
                "underlying_price",
                "mark_iv",
                "Time to Expiration",
                "Option Type",
                "Expiration Date"
            ]
        )

        if data.empty:
            st.warning("All rows were removed after cleaning the data.")
            st.stop()

        # ─────────────────────────────────────
        # BVIX DISPLAY
        # ─────────────────────────────────────

        st.subheader(f"📊 {coin} Volatility Index")

        with st.spinner("Calculating VIX-style crypto volatility index..."):
            bvix = calculate_crypto_vix(data, interest_rate)

        if bvix is not None:
            st.metric(
                label=f"{coin} 30-Day Implied Volatility Index",
                value=f"{bvix:.2f}",
                help=(
                    "Calculated using a VIX-style variance swap replication formula "
                    "applied to Deribit options data."
                )
            )

        else:
            st.warning("Could not compute the volatility index because there was insufficient valid option data.")

        st.write("---")

        # ─────────────────────────────────────
        # STRIKE FILTER
        # ─────────────────────────────────────

        min_strike, max_strike = strike_range

        data = data[
            (data["Strike Price"] >= data["underlying_price"] * min_strike) &
            (data["Strike Price"] <= data["underlying_price"] * max_strike)
        ]

        if data.empty:
            st.warning("No options left after applying the strike range filter.")
            st.stop()

        # ─────────────────────────────────────
        # IMPLIED VOL CALCULATION
        # ─────────────────────────────────────

        results = []

        for _, row in data.iterrows():
            S = row["underlying_price"]
            K = row["Strike Price"]
            T = row["Time to Expiration"]
            market_price = row["mark_price"]
            initial_vol = row["mark_iv"] / 100

            option_type = "call" if row["Option Type"] == "Call" else "put"

            iv = implied_volatility(
                market_price=market_price,
                S=S,
                K=K,
                T=T,
                r=interest_rate,
                initial_vol=initial_vol,
                option_type=option_type
            )

            results.append(iv)

        data["BSM_implied_volatility"] = results

        data = data.dropna(subset=["BSM_implied_volatility"])

        if len(data) < 4:
            st.warning("Not enough valid implied volatility points to generate a surface plot.")
            st.dataframe(data)
            st.stop()

        # ─────────────────────────────────────
        # SURFACE PLOT DATA
        # ─────────────────────────────────────

        strikes = data["Strike Price"].values
        times_to_expiration = data["Time to Expiration"].values
        implied_vols = data["BSM_implied_volatility"].values * 100

        if len(np.unique(strikes)) < 2 or len(np.unique(times_to_expiration)) < 2:
            st.warning(
                "Not enough unique strikes or expirations to create a 3D surface. "
                "Try using the monthly settlement period or widening the strike range."
            )
            st.dataframe(data)
            st.stop()

        num_points = 100

        fine_strikes = np.linspace(
            strikes.min(),
            strikes.max(),
            num_points
        )

        fine_times_to_expiration = np.linspace(
            times_to_expiration.min(),
            times_to_expiration.max(),
            num_points
        )

        X_fine, Y_fine = np.meshgrid(
            fine_strikes,
            fine_times_to_expiration
        )

        Z_fine = griddata(
            points=(strikes, times_to_expiration),
            values=implied_vols,
            xi=(X_fine, Y_fine),
            method="linear"
        )

        if np.isnan(Z_fine).all():
            Z_fine = griddata(
                points=(strikes, times_to_expiration),
                values=implied_vols,
                xi=(X_fine, Y_fine),
                method="nearest"
            )

        # ─────────────────────────────────────
        # OPENAI STRATEGY SECTION
        # This runs before the expander so strategies exists,
        # but the expander is displayed after the plot.
        # ─────────────────────────────────────

        prompt = f"""
You are an experienced quantitative analyst at a crypto options trading desk.

Analyze the following options data, which will be used to generate an implied volatility surface plot.

Settlement period: {settlement_per}
Coin: {coin}

Consider volatility trends, expiration dates, strike prices, and option type.

Also run basic no-arbitrage checks such as calendar spread and butterfly arbitrage.
Detect any obvious violations.

Suggest hedging or speculative approaches suitable for different market conditions.

{coin} Options Data with Strike Price, Time to Expiration, BSM_implied_volatility, and Option Type:
{data[["Strike Price", "Time to Expiration", "BSM_implied_volatility", "Option Type"]].to_string(index=False)}

Only give the top 2 strategies. Keep the whole output under 200 tokens.
"""

        try:
            if not openai_api_key:
                strategies = "OpenAI API key not found. Set OPENAI_API_KEY to generate strategy suggestions."
            else:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant and quantitative analyst."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.7
                )

                strategies = response.choices[0].message.content.strip()

        except Exception as e:
            strategies = f"OpenAI API error: {e}"

        # ─────────────────────────────────────
        # PLOTLY SURFACE PLOT
        # This is now BEFORE the two expanders.
        # ─────────────────────────────────────

        fig = go.Figure(
            data=[
                go.Surface(
                    z=Z_fine,
                    x=X_fine,
                    y=Y_fine,
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="I.V. %")
                )
            ]
        )

        fig.update_layout(
            title="Implied Volatility Surface",
            autosize=False,
            width=750,
            height=750,
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Time to Expiry (Years)",
                zaxis_title="Implied Volatility %",
                xaxis=dict(type="log"),
                aspectmode="cube"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("Valid IV points for surface:", len(data))

        # ─────────────────────────────────────
        # CALCULATED DATA EXPANDER
        # ─────────────────────────────────────

        with st.expander("View calculated options data"):
            st.dataframe(data)

        # ─────────────────────────────────────
        # STRATEGY EXPANDER
        # ─────────────────────────────────────

        with st.expander("Recommended Trading Strategies"):
            st.markdown(f"### Trading Strategies for {coin.upper()} Options")
            st.write(strategies)


st.write("---")

st.markdown(
    "Created by Ethan Falcao | "
    "[LinkedIn](https://www.linkedin.com/in/ethan-falcao/)"
)