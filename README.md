# Cryptocurrency Options Implied Volatility Surface Calculator

![Application Demo](.gif/ivf_defi.gif)

### Project Overview

This repository contains an interactive Streamlit web application that calculates and visualizes implied volatility surfaces for cryptocurrency options. The application uses real-time on-chain data from Deribit for over 700 options and applies a Black-Scholes-based model to compute implied volatility with the Newton-Raphson method, leveraging Vega for iterative convergence. Users can adjust three parameters and view the implied volatility surface through a 3D Plotly plot.

### Features

- **Real-time Data**: Retrieves data for 700+ cryptocurrency options using the Deribit API.
- **Advanced Volatility Calculations**: Implements a Black-Scholes model with Newton-Raphson and Vega for iterative convergence on highly volatile assets.
- **Interactive 3D Visualization**: Provides a customizable 3D surface plot for implied volatility across multiple parameters.

### Repository Structure

```plaintext
.
├── .devcontainer/                # Development container setup files
├── data/                         # Contains data and plots generated in the notebooks
├── 1. Data-Preprocessing.ipynb   # Preprocessing steps for data cleaning and preparation
├── 2. (BTC) - Exploratory Data Analysis.ipynb  # EDA for Bitcoin options with Vega approach
├── 2. (ETH) - Exploratory Data Analysis.ipynb  # EDA for Ethereum options with updated IV surface colors
├── 3. implied_volatility_surface.ipynb # Notebook for calculating and visualizing implied volatility surface
├── README.md                     # Project documentation (this file)
├── requirements.txt              # Dependencies needed to run the application
└── vs.py                         # Main Streamlit application file


### Setup Instructions

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <repository-name>
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
