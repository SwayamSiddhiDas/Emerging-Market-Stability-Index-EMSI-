import streamlit as st
import pandas as pd
import altair as alt
import os

def main():
    """
    Main function to create the Streamlit dashboard.
    """
    st.set_page_config(layout="wide")
    st.title("Emerging Market Stability Index")

    # Load pre-computed data
    try:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        index_data = pd.read_csv(os.path.join(data_dir, 'index_data.csv'))
        stock_screener_data = pd.read_csv(os.path.join(data_dir, 'stock_screener_data.csv'))
        full_stability_index = pd.read_csv(os.path.join(data_dir, 'full_stability_index.csv'), index_col=0, parse_dates=True)
        featured_data = pd.read_csv(os.path.join(data_dir, 'featured_dataset.csv'), index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error("Data files not found. Please run the backend script to generate the data.")
        return

    # Country selection
    countries = ["Brazil", "India", "South Africa", "Poland"]
    country_map = {
        "Brazil": "BRA",
        "India": "IND",
        "South Africa": "ZAF",
        "Poland": "POL"
    }
    stock_lists = {
        "BRA": ["VALE", "PETR4.SA", "ITUB", "BBDC4.SA", "ABEV3.SA"],
        "IND": ["RELIANCE.NS", "TCS.NS", "HDB", "INFY", "HINDUNILVR.NS"],
        "ZAF": ["NPN.JO", "BHP", "CFR.JO", "ANH.JO", "FSR.JO"],
        "POL": ["PKO.WA", "PZU.WA", "CDR.WA", "LPP.WA", "DNP.WA"]
    }
    selected_country_name = st.sidebar.selectbox("Select a country", countries)
    selected_country_code = country_map[selected_country_name]

    # Filter data for the selected country
    country_index_data = index_data[index_data['country_code'] == selected_country_code].iloc[0]
    country_screener_data = stock_screener_data[stock_screener_data['country'] == selected_country_code]
    country_stability_history = full_stability_index[[selected_country_code]]


    # --- Summary Metrics ---
    st.header("Country Investment Prospects")
    st.dataframe(index_data[['country_code', 'stability_index', 'outlook']])

    # --- Stability Index Chart ---
    st.header(f"Stability Index for {selected_country_name}")
    
    # Ensure the index has a name before resetting it
    country_stability_history.index.name = 'Date'
    
    chart = alt.Chart(country_stability_history.reset_index()).mark_line().encode(
        x='Date:T',
        y=alt.Y(f'{selected_country_code}:Q', title='Stability Index')
    ).properties(
        title=f"Stability Index for {selected_country_name}"
    )
    st.altair_chart(chart, use_container_width=True)


    # --- Dynamic Outlook Meter ---
    st.header("Dynamic Outlook Meter")
    st.metric("Market Outlook", country_index_data['outlook'])

    # Key Drivers
    st.subheader("Key Drivers")

    positive_drivers = {
        country_index_data['positive_driver_1_name']: country_index_data['positive_driver_1_value'],
        country_index_data['positive_driver_2_name']: country_index_data['positive_driver_2_value'],
        country_index_data['positive_driver_3_name']: country_index_data['positive_driver_3_value']
    }
    negative_drivers = {
        country_index_data['negative_driver_1_name']: country_index_data['negative_driver_1_value'],
        country_index_data['negative_driver_2_name']: country_index_data['negative_driver_2_value'],
        country_index_data['negative_driver_3_name']: country_index_data['negative_driver_3_value']
    }

    col1, col2 = st.columns(2)
    with col1:
        st.success("Positive Drivers")
        for name, value in positive_drivers.items():
            st.markdown(f"- **{name}:** `{value:.4f}`")
    with col2:
        st.warning("Negative Drivers")
        for name, value in negative_drivers.items():
            st.markdown(f"- **{name}:** `{value:.4f}`")

    # --- Sector Performance ---
    st.header("Sector Performance (Cumulative Returns)")
    sector_tickers = stock_lists[selected_country_code]
    sector_data = featured_data[sector_tickers]
    sector_data.index = pd.to_datetime(sector_data.index, errors='coerce', format='%Y-%m-%d %H:%M:%S') # Ensure index is DatetimeIndex
    daily_returns = 1 + sector_data.pct_change()
    monthly_returns = daily_returns.resample('ME').prod()
    cumulative_monthly_returns = monthly_returns.cumprod() - 1
    st.line_chart(cumulative_monthly_returns.dropna())


    # --- Stability-Adjusted Stock Screener ---
    st.header("Stability-Adjusted Stock Screener")
    
    if not country_screener_data.empty:
        high_momentum = country_screener_data[country_screener_data['category'] == 'High-Momentum Play']
        resilient_plays = country_screener_data[country_screener_data['category'] == 'Resilient Defender']
        neutral = country_screener_data[country_screener_data['category'] == 'Neutral']

        tab1, tab2, tab3 = st.tabs(["High-Momentum Play", "Resilient Defender", "Neutral"])

        with tab1:
            st.dataframe(high_momentum[['ticker', 'company_name', 'correlation_to_index']])
        with tab2:
            st.dataframe(resilient_plays[['ticker', 'company_name', 'correlation_to_index']])
        with tab3:
            st.dataframe(neutral[['ticker', 'company_name', 'correlation_to_index']])
    else:
        st.warning("No stock screener data available for the selected country.")


if __name__ == "__main__":
    main()
