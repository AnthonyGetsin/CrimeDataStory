# safety_score.py (improved)
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from utils import fetch_crime_data, geocode_addresses, process_spatial_data

def calculate_crime_density_scores(crime_counts):
    """Calculate scores using percentile-based exponential decay"""
    # Calculate robust percentiles
    p25 = crime_counts.quantile(0.25)
    p50 = crime_counts.quantile(0.50)
    p75 = crime_counts.quantile(0.75)
    p90 = crime_counts.quantile(0.90)
    p95 = crime_counts.quantile(0.95)
    
    # Create non-linear scoring using sigmoid function
    x = crime_counts.clip(upper=p95)  # Cap at 95th percentile
    scaled = (x - p50) / (p95 - p50 + 1e-9)  # Normalize around median
    scores = 10 * (1 - 1 / (1 + np.exp(-3 * scaled)))  # Sigmoid with steep drop
    
    # Force distribution
    return np.round(scores, 1).clip(0.5, 9.5)

def precompute_safety_scores():
    print("🚀 Computing realistic safety scores...")
    
    # 1. Fetch extended historical data
    raw_df = fetch_crime_data(
        start_date=datetime(2016, 1, 1),
        end_date=datetime(2024, 12, 31)  # Actual available data
    )
    
    # 2. Process data
    geo_df = geocode_addresses(raw_df)
    geo_df, gdf_edges = process_spatial_data(geo_df)

    # 3. Calculate street-level metrics
    edge_counts = geo_df['edge'].value_counts().reset_index()
    edge_counts.columns = ['edge', 'crime_count']
    edge_counts[['u', 'v', 'key']] = pd.DataFrame(
        edge_counts['edge'].tolist(), 
        index=edge_counts.index
    )

    # 4. Advanced scoring
    edge_counts['safety_score'] = calculate_crime_density_scores(
        edge_counts['crime_count']
    )
    
    # 5. Merge and save
    streets_gdf = gdf_edges.reset_index()[['u', 'v', 'key', 'geometry']]
    streets_gdf = streets_gdf.merge(
        edge_counts,
        on=['u', 'v', 'key'],
        how='left'
    ).fillna({'safety_score': 10})  # Only completely safe streets get 10
    
    # Force normal distribution
    scores = streets_gdf['safety_score']
    streets_gdf['safety_score'] = np.round(
        (scores - scores.mean()) / scores.std() * 2 + 5, 1
    ).clip(0, 10)
    
    streets_gdf.to_file("realistic_safety_scores.geojson", driver='GeoJSON')
    print("✅ Realistic scores saved with distribution:")
    print(streets_gdf['safety_score'].describe())

if __name__ == "__main__":
    precompute_safety_scores()