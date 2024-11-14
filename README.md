### Project Overview

This repository contains an interactive Streamlit web application that calculates and visualizes implied volatility surfaces for cryptocurrency options. The application uses real-time on-chain data from Deribit for over 700 options and applies a Black-Scholes-based model to compute implied volatility with the Newton-Raphson method, leveraging Vega for iterative convergence. Users can adjust three parameters and view the implied volatility surface through a 3D Plotly plot, then connects to the GBT-4o-Mini model via API to analyze and generate optimal trading strategies based on real-time volatility trends. 


<div align="center">
    <img src="https://github.com/user-attachments/assets/0ab77740-7273-4d24-8353-cfef5ad692f1"  alt="IVF Defi GIF" width="600" height="400">
</div>


### Features

- **Real-time Data**: Retrieves data for 700+ cryptocurrency options using the Deribit API.
- **Advanced Volatility Calculations**: Implements a Black-Scholes model with Newton-Raphson and Vega for iterative convergence on highly volatile assets.
- **Interactive 3D Visualization**: Provides a customizable 3D surface plot for implied volatility across multiple parameters.
- **Trading Strategies for BTC Option**: Trading strategies generated from real-time data using OpenAI's GPT-4 Mini Model.

### Repository Structure

### Notebooks

- **1. Data-Preprocessing.ipynb**: Contains the data cleaning and preprocessing steps required for preparing the on-chain options data.
- **2. (BTC) - Exploratory Data Analysis.ipynb**: Conducts exploratory data analysis for Bitcoin options, focusing on the Vega-based approach for implied volatility.
- **3. implied_volatility_surface.ipynb**: This notebook calculates the implied volatility surface using a Black-Scholes-based model and visualizes it with Plotly.
- **4. vs.py**: Queries data, calculates implied volatility, and visualizes the volatility surface, hosting the results interactively on Streamlit.

### Formulas Used

#### 1. **Black-Scholes Model for Option Pricing**

The Black-Scholes formula is used to price European-style options and forms the basis for implied volatility calculation. For a call option, the Black-Scholes price $C$ is given by:

$$
C = S_0 \cdot N(d_1) - X \cdot e^{-rT} \cdot N(d_2)
$$

where:
- $S_0$ = current price of the underlying asset
- $X$ = strike price of the option
- $T$ = time to maturity of the option (in years)
- $r$ = risk-free interest rate
- $N(\cdot)$ = cumulative distribution function of the standard normal distribution

The terms $d_1$ and $d_2$ are calculated as:

$$
d_1 = \frac{\ln\left(\frac{S_0}{X}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T}
$$

For put options, the formula changes slightly to:

$$
P = X \cdot e^{-rT} \cdot N(-d_2) - S_0 \cdot N(-d_1)
$$

#### 2. **Implied Volatility and the Newton-Raphson Method**

Implied volatility ($\sigma_{\text{impl}}$) is the value of $\sigma$ that matches the observed option price $C_{\text{obs}}$. The Newton-Raphson method approximates it iteratively:

$$
\sigma_{\text{new}} = \sigma_{\text{old}} - \frac{f(\sigma_{\text{old}})}{f'(\sigma_{\text{old}})}
$$

where:
- $f(\sigma) = C(\sigma) - C_{\text{obs}}$
- $f'(\sigma)$ = Vega (partial derivative of the option price with respect to volatility)

#### 3. **Vega Calculation**

Vega, used in iterative convergence, is given by:

$$
\text{Vega} = S_0 \cdot \sqrt{T} \cdot N'(d_1)
$$

where $N'(d_1)$ is the probability density function of the standard normal distribution evaluated at $d_1$.

#### 4. **Implied Volatility Adjustment Using Newton-Raphson**

Using Vega, we adjust volatility $\sigma$ iteratively until the calculated option price $C(\sigma)$ closely matches $C_{\text{obs}}$:

$$
\sigma_{\text{new}} = \sigma_{\text{old}} - \frac{C(\sigma_{\text{old}}) - C_{\text{obs}}}{\text{Vega}}
$$

### Usage

- **Calculation & Visualization**: After running the app, use the interactive interface to input parameters and generate the implied volatility surface.
