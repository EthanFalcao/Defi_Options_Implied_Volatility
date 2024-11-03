# Cryptocurrency Options Implied Volatility Surface Calculator

<div align="center">
    <img src="https://github.com/user-attachments/assets/166a6c35-75fe-4d39-bc88-d089344aa1f7"  alt="IVF Defi GIF" width="600" height="400">
</div>


 


### Project Overview


This repository contains an interactive Streamlit web application that calculates and visualizes implied volatility surfaces for cryptocurrency options. The application uses real-time on-chain data from Deribit for over 700 options and applies a Black-Scholes-based model to compute implied volatility with the Newton-Raphson method, leveraging Vega for iterative convergence. Users can adjust three parameters and view the implied volatility surface through a 3D Plotly plot.

### Features

- **Real-time Data**: Retrieves data for 700+ cryptocurrency options using the Deribit API.
- **Advanced Volatility Calculations**: Implements a Black-Scholes model with Newton-Raphson and Vega for iterative convergence on highly volatile assets.
- **Interactive 3D Visualization**: Provides a customizable 3D surface plot for implied volatility across multiple parameters.

### Repository Structure

```plaintext

1. **Clone the Repository**:
   
   git clone <repository-url>
   cd <repository-name>
```
   ```

2. **Install Required Packages**:
   Use the following command to install the necessary packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Streamlit application using:
   ```
   streamlit run vs.py
   ```

### Notebooks

- **1. Data-Preprocessing.ipynb**: Contains the data cleaning and preprocessing steps required for preparing the on-chain options data.
- **2. (BTC) - Exploratory Data Analysis.ipynb**: Conducts exploratory data analysis for Bitcoin options, focusing on the Vega-based approach for implied volatility.
- **2. (ETH) - Exploratory Data Analysis.ipynb**: Similar to the Bitcoin EDA, but focused on Ethereum options with customized colors (green and red) for the IV surface.
- **3. implied_volatility_surface.ipynb**: This notebook calculates the implied volatility surface using a Black-Scholes-based model and visualizes it with Plotly.

### Usage

- **Calculation & Visualization**: After running the app, use the interactive interface to input parameters and generate the implied volatility surface.
- **GIF Preview**: The application demonstration GIF is included above as a quick visual guide. Place the `test.gif` file in the repository root folder.
