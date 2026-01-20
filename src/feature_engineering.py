import pandas as pd
from transformers import pipeline
import os

def calculate_sentiment(df):
    """
    Calculates a daily sentiment score from news text using FinBERT.
    """
    # Placeholder for sentiment analysis
    print("Calculating sentiment...")
    # sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    # df['sentiment'] = df['news_text'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    return df

def calculate_volatility(df):
    """
    Calculates the 30-day rolling volatility for stock and currency markets.
    """
    print("Calculating volatility...")
    for col in df.columns:
        if col.startswith('^') or col.endswith('=X'):
            df[f'{col}_volatility'] = df[col].pct_change().rolling(window=30).std() * (252**0.5)
    return df

def main():
    """
    Main function to orchestrate the feature engineering pipeline.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_path = os.path.join(data_dir, 'unified_dataset.csv')
    output_path = os.path.join(data_dir, 'featured_dataset.csv')
    
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Calculate sentiment (currently disabled)
    # df = calculate_sentiment(df)
    
    # Calculate volatility
    df = calculate_volatility(df)
    
    df.to_csv(output_path)
    print("Feature engineering complete.")

if __name__ == "__main__":
    main()
