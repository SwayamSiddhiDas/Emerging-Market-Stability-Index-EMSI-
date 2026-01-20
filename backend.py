import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import wbdata
import os

# STEP 1: Verify Stock Universe Definition
STOCK_UNIVERSE = {
    'BRA': [("VALE", "Vale"), ("PETR4.SA", "Petrobras"), ("ITUB", "Itaú Unibanco"), ("BBDC4.SA", "Bradesco"), ("ABEV3.SA", "Ambev"), ("WEGE3.SA", "WEG"), ("MGLU3.SA", "Magazine Luiza"), ("LREN3.SA", "Lojas Renner"), ("RENT3.SA", "Localiza"), ("GGBR4.SA", "Gerdau")],
    'IND': [("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy"), ("HDB", "HDFC Bank"), ("INFY", "Infosys"), ("HINDUNILVR.NS", "Hindustan Unilever"), ("ICICIBANK.NS", "ICICI Bank"), ("KOTAKBANK.NS", "Kotak Mahindra Bank"), ("SBIN.NS", "State Bank of India"), ("BAJFINANCE.NS", "Bajaj Finance"), ("BHARTIARTL.NS", "Bharti Airtel")],
    'ZAF': [("NPN.JO", "Naspers"), ("BHP", "BHP Group"), ("CFR.JO", "Compagnie Financière Richemont"), ("ANH.JO", "Anheuser-Busch InBev"), ("FSR.JO", "FirstRand"), ("GLN.JO", "Glencore"), ("SBK.JO", "Standard Bank"), ("VOD.JO", "Vodacom"), ("MTN.JO", "MTN Group"), ("SOL.JO", "Sasol")],
    'POL': [("PKO.WA", "PKO Bank Polski"), ("PZU.WA", "PZU"), ("CDR.WA", "CD Projekt"), ("LPP.WA", "LPP"), ("DNP.WA", "Dino Polska"), ("KGH.WA", "KGHM Polska Miedź"), ("MBK.WA", "mBank"), ("PEO.WA", "Bank Pekao"), ("SPL.WA", "Santander Bank Polska"), ("TPE.WA", "Tauron Polska Energia")]
}

def fetch_world_bank_data(start_date, end_date, country_codes):
    print("Fetching World Bank data...")
    indicators = {'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)', 'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)'}
    return wbdata.get_dataframe(indicators, country=country_codes, date=(pd.to_datetime(start_date), pd.to_datetime(end_date)))

def fetch_yfinance_data(start_date, end_date, tickers):
    print("Fetching Yahoo Finance data...")
    return yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

def calculate_volatility(df):
    print("Calculating volatility...")
    for col in df.columns:
        if isinstance(col, str) and (col.startswith('^') or col.endswith('=X') or '.' in col or col.isupper()):
            df[f'{col}_volatility'] = df[col].pct_change().rolling(window=30).std() * (252**0.5)
    return df

def preprocess_data(df):
    df = df.fillna(method='ffill').dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

def build_index(df):
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df)
    index = pd.Series(principal_components.flatten(), index=df.index, name='stability_index')
    loadings = pd.Series(pca.components_[0], index=df.columns)
    return index, loadings

# STEP 2: Fix Stock Data Fetching
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            print(f"  ⚠ {ticker}: No data returned from Yahoo Finance")
            return None
        if len(hist) < 30:
            print(f"  ⚠ {ticker}: Only {len(hist)} days of data (minimum 30 required)")
            return None
        print(f"  [OK] {ticker}: Fetched {len(hist)} days of data")
        return hist
    except Exception as e:
        print(f"  [ERROR] {ticker}: {str(e)}")
        return None

# STEP 3: Fix Index Data Preparation
def prepare_index_data(index_data):
    index_data['date'] = pd.to_datetime(index_data['date'])
    index_data = index_data.sort_values('date').reset_index(drop=True)
    index_data['index_change'] = index_data['stability_index'].pct_change()
    index_data = index_data.dropna(subset=['index_change'])
    print(f"Index prepared: {len(index_data)} days from {index_data['date'].min()} to {index_data['date'].max()}")
    return index_data

# STEP 4: Fix Date Alignment & Correlation Calculation
def calculate_stock_correlation(stock_hist, index_data, ticker):
    stock_df = stock_hist[['Close']].copy()
    if stock_df.index.tz is not None:
        stock_df.index = stock_df.index.tz_localize(None)
    stock_df['stock_return'] = stock_df['Close'].pct_change()
    stock_df = stock_df.reset_index()
    stock_df.columns = ['date', 'close', 'stock_return']
    
    merged = pd.merge(index_data[['date', 'index_change']], stock_df[['date', 'stock_return']], on='date', how='inner')
    merged = merged.dropna()
    
    print(f"  Aligned data points: {len(merged)} days")
    if len(merged) < 30:
        print(f"  ⚠ Insufficient aligned data (need 30+, have {len(merged)})")
        return None, merged
        
    correlation = merged['index_change'].corr(merged['stock_return'])
    print(f"  Correlation: {correlation:.3f}")
    return correlation, merged

# STEP 5: Fix Stock Categorization Logic
def categorize_stock(correlation, merged_data):
    if correlation is None or pd.isna(correlation):
        return None, {}
    
    stock_returns = merged_data['stock_return'].tail(30)
    avg_return_30d = stock_returns.mean() * 100
    volatility_30d = stock_returns.std() * 100
    
    category = None
    if correlation > 0.02:
        category = "High-Momentum Play"
    elif correlation < -0.02:
        category = "Resilient Defender"
        
    metrics = {'avg_return_30d': round(avg_return_30d, 2), 'volatility_30d': round(volatility_30d, 2), 'data_points': len(merged_data)}
    return category, metrics

# STEP 6: Add Comprehensive Debug Logging
def process_stocks_for_country(country_code, index_data, start_date, end_date):
    print(f"\n{'='*60}\nPROCESSING COUNTRY: {country_code}\n{'='*60}")
    stocks = STOCK_UNIVERSE.get(country_code, [])
    print(f"Total stocks to process: {len(stocks)}")
    
    results = []
    skipped = {'no_data': 0, 'insufficient_alignment': 0, 'weak_correlation': 0}
    
    for ticker, company_name in stocks:
        print(f"\n--- {ticker} ({company_name}) ---")
        stock_hist = fetch_stock_data(ticker, start_date, end_date)
        if stock_hist is None:
            skipped['no_data'] += 1
            continue
        
        correlation, merged = calculate_stock_correlation(stock_hist, index_data, ticker)
        if correlation is None:
            skipped['insufficient_alignment'] += 1
            continue
            
        category, metrics = categorize_stock(correlation, merged)
        if category is None:
            print(f"  [INFO] Excluded: Weak correlation ({correlation:.3f})")
            skipped['weak_correlation'] += 1
            continue
            
        print(f"  [SUCCESS] Categorized as: {category}")
        results.append({'country': country_code, 'ticker': ticker, 'company_name': company_name, 'correlation_to_index': round(correlation, 3), 'category': category, **metrics})
        
    print(f"\n{'='*60}\nSUMMARY FOR {country_code}\n{'='*60}")
    print(f"[SUCCESS] Successfully categorized: {len(results)}")
    print(f"   - High-Momentum Plays: {sum(1 for r in results if r['category']=='High-Momentum Play')}")
    print(f"   - Resilient Defenders: {sum(1 for r in results if r['category']=='Resilient Defender')}")
    print(f"[INFO] Skipped:\n   - No data available: {skipped['no_data']}\n   - Insufficient alignment: {skipped['insufficient_alignment']}\n   - Weak correlation: {skipped['weak_correlation']}")
    return results

# STEP 7: Fix CSV Output Generation
def save_stock_screener_data(all_results, output_file='stock_screener_data.csv'):
    if not all_results:
        print("[ERROR] No results to save.")
        return False
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, output_file)
    
    screener_df = pd.DataFrame(all_results)
    screener_df = screener_df.sort_values(['country', 'category', 'correlation_to_index'], ascending=[True, True, False])
    screener_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}\nFINAL OUTPUT\n{'='*60}")
    print(f"📁 Saved to: {output_path}\n📊 Total stocks: {len(screener_df)}\n\nBreakdown by country:")
    for country in ['BRA', 'IND', 'ZAF', 'PL']:
        country_df = screener_df[screener_df['country'] == country]
        momentum = len(country_df[country_df['category'] == 'High-Momentum Play'])
        defender = len(country_df[country_df['category'] == 'Resilient Defender'])
        print(f"  {country}: {len(country_df)} total (Momentum: {momentum}, Defender: {defender})")
    return True

def main():
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    countries = {"BRA": "BR", "IND": "IN", "ZAF": "ZA", "POL": "PL"}
    stock_indices = ["^BVSP", "^NSEI", "^J203.JO", "^WIG20"]
    currencies = ["BRLUSD=X", "INRUSD=X", "ZARUSD=X", "PLNUSD=X"]
    
    # --- Part 1: Index Computation (largely unchanged) ---
    all_tickers = stock_indices + currencies
    yfinance_data = fetch_yfinance_data(start_date, end_date, all_tickers)
    world_bank_data = fetch_world_bank_data(start_date, end_date, list(countries.keys()))
    world_bank_data = world_bank_data.unstack(level=0)
    master_df = pd.concat([world_bank_data, yfinance_data], axis=1)
    featured_df = calculate_volatility(master_df.copy())
    
    index_data_rows = []
    all_indices = []
    for i, country_code in enumerate(countries.keys()):
        index_cols = [col for col in featured_df.columns if (stock_indices[i] in str(col) or currencies[i] in str(col))]
        country_df = featured_df[index_cols]
        processed_df = preprocess_data(country_df)
        stability_index, loadings = build_index(processed_df)
        stability_index.name = country_code
        all_indices.append(stability_index)
        
        vol_7d = stability_index.pct_change().rolling(window=7).std().iloc[-1]
        vol_14d = stability_index.pct_change().rolling(window=14).std().iloc[-1]
        outlook = "Neutral"
        if vol_7d > vol_14d * 1.5: outlook = "Cautious"
        elif vol_7d < vol_14d * 0.75: outlook = "Bullish"
        
        top_positive = loadings.nlargest(3)
        top_negative = loadings.nsmallest(3)
        
        def clean_feature_name(name):
            name = str(name).replace("_volatility", " Volatility").replace("('", "").replace("', '", " (").replace("')", ")")
            return name.title()

        latest_index_data = {'date': stability_index.index[-1], 'country_code': country_code, 'stability_index': stability_index.iloc[-1], 'outlook': outlook}
        for j in range(3):
            latest_index_data[f'positive_driver_{j+1}_name'] = clean_feature_name(top_positive.index[j])
            latest_index_data[f'positive_driver_{j+1}_value'] = top_positive.values[j]
            latest_index_data[f'negative_driver_{j+1}_name'] = clean_feature_name(top_negative.index[j])
            latest_index_data[f'negative_driver_{j+1}_value'] = top_negative.values[j]
        index_data_rows.append(latest_index_data)

    full_index_history = pd.concat(all_indices, axis=1).reset_index().rename(columns={'index': 'date'})
    
    # Create the data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    full_index_history.to_csv(os.path.join(data_dir, 'full_stability_index.csv'), index=False)
    pd.DataFrame(index_data_rows).to_csv(os.path.join(data_dir, 'index_data.csv'), index=False)
    
    # --- Part 2: Stock Screener Computation ---
    all_results = []
    for country_code in countries.keys():
        country_index_df = full_index_history[['date', country_code]].rename(columns={country_code: 'stability_index'})
        prepared_index_df = prepare_index_data(country_index_df)
        country_results = process_stocks_for_country(country_code, prepared_index_df, start_date, end_date)
        all_results.extend(country_results)
    
    # Add this line to ensure all stocks are included in the yfinance call
    all_stock_tickers = [ticker for stock_list in STOCK_UNIVERSE.values() for ticker, name in stock_list]
    all_tickers.extend(all_stock_tickers)
    tickers = list(set(all_tickers))
        
    save_stock_screener_data(all_results)
    print("Backend pre-computation complete.")

if __name__ == "__main__":
    main()