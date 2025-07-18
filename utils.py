# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd
import requests
import json
from datetime import datetime
from pathlib import Path
from geopy.geocoders import Nominatim
import time
from geopy.extra.rate_limiter import RateLimiter

@st.cache_data
def fetch_crime_data(start_date, end_date):
    with st.status("🔍 Fetching crime data...", expanded=True) as status:
        all_features = []
        offset = 0
        batch_size = 2000
        
        base_url = "https://services7.arcgis.com/vIHhVXjE1ToSg0Fz/arcgis/rest/services/Berkeley_PD_Cases_2016_to_Current/FeatureServer/0/query"
        where_clause = (
            f"Occurred_Datetime >= DATE '{start_date.strftime('%Y-%m-%d')} 00:00:00' "
            f"AND Occurred_Datetime < DATE '{end_date.strftime('%Y-%m-%d')} 00:00:00'"
        )

        params = {
            "where": where_clause,
            "outFields": "Incident_Type,Block_Address,Occurred_Datetime",
            "outSR": "4326",
            "f": "json",
            "returnGeometry": "false"
        }

        while True:
            params.update({"resultOffset": offset, "resultRecordCount": batch_size})
            response = requests.get(base_url, params=params)
            data = response.json()
            features = data.get("features", [])
            if not features:
                break
            all_features.extend(features)
            offset += batch_size
            status.update(label=f"Fetched {len(all_features)} records...", state="running")
            time.sleep(0.1)

        df = pd.DataFrame([feature.get("attributes", {}) for feature in all_features])
        df["Occurred_Datetime"] = pd.to_datetime(df["Occurred_Datetime"], unit="ms")
        return df

@st.cache_data
def geocode_addresses(_df):
    with st.status("🌍 Geocoding...", expanded=True) as status:
        cache_file = Path("geocode_cache.json")
        geocode_cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        
        geolocator = Nominatim(user_agent="crime_map")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        unique_addresses = _df['Block_Address'].dropna().unique()
        address_coords = {}
        
        progress_bar = st.progress(0)
        for i, address in enumerate(unique_addresses):
            if address not in geocode_cache:
                try:
                    location = geocode(f"{address}, Berkeley, CA", timeout=10)
                    geocode_cache[address] = (location.latitude, location.longitude) if location else None
                except:
                    geocode_cache[address] = None
            address_coords[address] = geocode_cache[address]
            progress_bar.progress((i+1)/len(unique_addresses))
        
        cache_file.write_text(json.dumps(geocode_cache))
        _df['coords'] = _df['Block_Address'].map(address_coords)
        _df = _df.dropna(subset=['coords'])
        _df[['latitude', 'longitude']] = pd.DataFrame(_df['coords'].tolist(), index=_df.index)
        return _df

def process_spatial_data(_df):
    with st.status("📡 Analyzing streets...", expanded=True) as status:
        G = ox.graph_from_place("Berkeley, California, USA", network_type='drive')
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        
        crime_points = gpd.GeoDataFrame(
            _df,
            geometry=gpd.points_from_xy(_df.longitude, _df.latitude),
            crs="EPSG:4326"
        ).to_crs(G.graph['crs'])
        
        X = crime_points.geometry.x.values
        Y = crime_points.geometry.y.values
        _df['edge'] = ox.distance.nearest_edges(G, X, Y)
        return _df, gdf_edges

def add_time_of_day(df):
    df['hour'] = df['Occurred_Datetime'].dt.hour
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )
    return df