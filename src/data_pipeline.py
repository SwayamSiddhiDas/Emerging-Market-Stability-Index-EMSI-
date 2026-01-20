import pandas as pd
import wbdata
import yfinance as yf
from google.cloud import bigquery
import os

# TODO: Set up Google Cloud credentials for BigQuery

def fetch_gdelt_data(start_date, end_date, country_code):
    """
    Fetches news and events data from the GDELT Project using BigQuery.
    """
    print(f"Fetching GDELT data for {country_code} from {start_date} to {end_date}...")
    client = bigquery.Client()
    query = """
        SELECT date, V2Tone
        FROM `gdelt-bq.gdeltv2.events`
        WHERE ActionGeo_CountryCode = @country_code
        AND SQLDATE BETWEEN @start_date AND @end_date
    """
    params = [
        bigquery.ScalarQueryParameter("country_code", "STRING", country_code),
        bigquery.ScalarQueryParameter("start_date", "INT64", int(start_date.replace('-', ''))),
        bigquery.ScalarQueryParameter("end_date", "INT64", int(end_date.replace('-', ''))),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    df = client.query(query, job_config=job_config).to_dataframe()
    return df

def fetch_world_bank_data(start_date, end_date, country_codes):
    """
    Fetches economic indicators from the World Bank.
    """
    print(f"Fetching World Bank data for {country_codes} from {start_date} to {end_date}...")
    indicators = {
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)'
    }
    df = wbdata.get_dataframe(
        indicators,
        country=country_codes,
        date=(pd.to_datetime(start_date), pd.to_datetime(end_date)),
    )
    return df

def fetch_yfinance_data(start_date, end_date, tickers):
    """
    Fetches market prices (stock indices and currencies) from Yahoo Finance.
    """
    print(f"Fetching Yahoo Finance data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    return data

def main():
    """
    Main function to orchestrate the data pipeline.
    """
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    # Country Portfolio
    countries = {
        "BRA": "BR", # Brazil
        "IND": "IN", # India
        "ZAF": "ZA", # South Africa
        "POL": "PL"  # Poland
    }

    # Yahoo Finance Tickers
    stock_indices = ["^BVSP", "^NSEI", "^J203.JO", "^WIG20"]
    currencies = ["BRLUSD=X", "INRUSD=X", "ZARUSD=X", "PLNUSD=X"]
    
    # Sector ETFs and Blue-Chip Stocks
    # Blue-Chip and Top Stocks
    stock_lists = {
        "BRA": ["VALE", "PETR4.SA", "ITUB", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA", "MGLU3.SA", "LREN3.SA", "RENT3.SA", "GGBR4.SA"],
        "IND": ["RELIANCE.NS", "TCS.NS", "HDB", "INFY", "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"],
        "ZAF": ["NPN.JO", "BHP", "CFR.JO", "ANH.JO", "FSR.JO", "GLN.JO", "SBK.JO", "VOD.JO", "MTN.JO", "SOL.JO"],
        "POL": ["PKO.WA", "PZU.WA", "CDR.WA", "LPP.WA", "DNP.WA", "KGH.WA", "MBK.WA", "PEO.WA", "SPL.WA", "TPE.WA"]
    }
    
    all_tickers = stock_indices + currencies
    for country_tickers in stock_lists.values():
        all_tickers.extend(country_tickers)
    
    tickers = list(set(all_tickers)) # Remove duplicates

    # Fetch data from all sources
    # gdelt_data = fetch_gdelt_data(start_date, end_date, "BR") # Example for one country
    world_bank_data = fetch_world_bank_data(start_date, end_date, list(countries.keys()))
    yfinance_data = fetch_yfinance_data(start_date, end_date, tickers)

    # TODO: Merge the datasets into a single, unified time-series dataset.
    # print("GDELT data shape:", gdelt_data.shape)
    print("World Bank data shape:", world_bank_data.shape)
    print("Yahoo Finance data shape:", yfinance_data.shape)

    # Placeholder for merging logic
    world_bank_data = world_bank_data.unstack(level=0)
    merged_df = pd.concat([world_bank_data, yfinance_data], axis=1)
    print("Merged data shape:", merged_df.shape)
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'unified_dataset.csv')
    merged_df.to_csv(output_path)

    print("Data pipeline script structure created.")


if __name__ == "__main__":
    main()
