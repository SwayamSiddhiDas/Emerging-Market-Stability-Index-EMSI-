import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_data(df):
    """
    Preprocesses the data for PCA by scaling and filling missing values.
    """
    print("Preprocessing data...")
    df = df.fillna(method='ffill').dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

def build_index(df):
    """
    Builds the stability index using PCA and identifies key drivers.
    """
    print("Building index...")
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(df)
    
    # Create the index series
    index = pd.Series(principal_components.flatten(), index=df.index, name='Stability_Index')
    
    # Get feature loadings to identify key drivers
    loadings = pd.Series(pca.components_[0], index=df.columns)
    
    return index, loadings

def main():
    """
    Main function to orchestrate the model building pipeline.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_path = os.path.join(data_dir, 'featured_dataset.csv')
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    country_map = {
        "BRA": ["^BVSP", "BRLUSD=X"],
        "IND": ["^NSEI", "INRUSD=X"],
        "ZAF": ["^J203.JO", "ZARUSD=X"],
        "POL": ["^WIG20", "PLNUSD=X"]
    }

    all_indices = []

    for country_code, assets in country_map.items():
        print(f"Building index for {country_code}...")
        
        # Filter columns for the specific country
        country_columns = [col for col in df.columns if any(asset in col for asset in assets)]
        country_df = df[country_columns]

        # Preprocess the data
        processed_df = preprocess_data(country_df)
        
        # Build the index
        stability_index, loadings = build_index(processed_df)
        stability_index.name = country_code # Rename the series
        all_indices.append(stability_index)
        
        # Save loadings for the dashboard
        loadings_output_path = os.path.join(data_dir, f'{country_code}_loadings.csv')
        loadings.to_csv(loadings_output_path)

    # Combine all indices into a single dataframe
    final_indices = pd.concat(all_indices, axis=1)
    final_indices_output_path = os.path.join(data_dir, 'stability_index.csv')
    final_indices.to_csv(final_indices_output_path)
    print("Index building complete for all countries.")

if __name__ == "__main__":
    main()