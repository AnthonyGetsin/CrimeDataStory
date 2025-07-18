import streamlit as st
import time
import requests
import pandas as pd
import folium
import hdbscan
from streamlit_folium import st_folium
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime
import osmnx as ox
import geopandas as gpd
import numpy as np
from geopy.geocoders import Nominatim
from pathlib import Path
import json
from collections import defaultdict
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN
import networkx as nx
from folium.plugins import HeatMap
from shapely.geometry import Point, LineString
from predictive_features import predictive_features_page
from utils import fetch_crime_data, add_time_of_day, process_spatial_data, geocode_addresses


# ============================================================================
# INITIAL SETUP
# ============================================================================
st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.title("🚨 Berkeley Crime Analysis")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'types': [],
        'times': ['Night', 'Morning', 'Afternoon', 'Evening'],
        'min_severity': 0
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================    
def create_crime_map(filtered_df, edge_counts, gdf_edges):
    """Create optimized Folium map with dynamic styling"""
    m = folium.Map(
        location=[37.87, -122.27], 
        zoom_start=14,
        tiles='CartoDB positron',
        prefer_canvas=True,  # Better for large datasets
        control_scale=True
    )

    folium.TileLayer(
        'CartoDB positron',
        name='Base Map',
        max_zoom=18,
        min_zoom=12,
        control=False,
        tile_size=256,
        zoom_interval=1
    ).add_to(m)

    def classify_counts(counts):
        if not counts:
            return []
        values = list(counts.values())
        return np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8, 1.0])).tolist()

    street_features = []
    breaks = classify_counts(edge_counts)
    
    for (u, v, k), count in edge_counts.items():
        edge = gdf_edges.loc[(u, v, k)]
        color = get_color_for_count(count, breaks)
        
        feature = {
            "type": "Feature",
            "geometry": edge.geometry.__geo_interface__,
            "properties": {
                "count": count,
                "color": color
            }
        }
        street_features.append(feature)

    folium.GeoJson(
        {"type": "FeatureCollection", "features": street_features},
        style_function=lambda x: {
            'color': x['properties']['color'],
            'weight': 6,
            'opacity': 0.7
        },
        name='street_crime_density',
        zoom_on_click=False,
        control=False
    ).add_to(m)

    marker_cluster = MarkerCluster(
        name="Crime Locations",
        options={
            'disableClusteringAtZoom': 21,
            'spiderfyOnMaxZoom': True,
            'showCoverageOnHover': True,
            'chunkedLoading': True,
            'chunkInterval': 100
        }
    ).add_to(m)

    for idx, row in filtered_df.iterrows():
        popup_html = f"""
            <div style="font-family: Arial; font-size: 12px">
                <strong>Type:</strong> {row['Incident_Type']}<br>
                <strong>Date:</strong> {row['Occurred_Datetime'].strftime('%Y-%m-%d %H:%M')}<br>
                <strong>Address:</strong> {row['Block_Address']}<br>
                <strong>Time of Day:</strong> {row['time_of_day']}
            </div>
        """
        
        marker_cluster.add_child(
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='#0074D9',
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250)
            )
        )
    
    m.add_child(folium.Element(
        """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const map = document.querySelector('.folium-map');
            map.addEventListener('click', function(e) {
                if (e.target.closest('.marker-cluster')) {
                    // Trigger Streamlit rerun when clusters are clicked
                    window.parent.document.dispatchEvent(
                        new CustomEvent('clusterClick', {detail: "clicked"})
                    );
                }
            });
        });
        </script>
        """
    ))

    return m

def get_color_for_count(count, breaks):
    color_palette = ['#2ECC40', '#FFDC00', '#FF851B', '#FF4136', '#85144b']
    if not breaks:
        return '#CCCCCC'
    for i in range(len(breaks)-1):
        if breaks[i] <= count < breaks[i+1]:
            return color_palette[i]
    return color_palette[-1]

# ============================================================================
# MAIN APP
# ============================================================================
def original_main():
    # Sidebar controls
    with st.sidebar:
        st.subheader("Filters")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime(2025, 1, 1))
        with col2:
            end_date = st.date_input("End date", datetime(2025, 4, 1))
        
        if st.button("🚀 Run Analysis", use_container_width=True):
            with st.spinner("Processing..."):
                raw_df = fetch_crime_data(start_date, end_date)
                if not raw_df.empty:
                    geo_df = geocode_addresses(raw_df)
                    geo_df = add_time_of_day(geo_df)
                    geo_df, gdf_edges = process_spatial_data(geo_df)
                    
                    st.session_state.processed_data = {
                        'geo_df': geo_df,
                        'gdf_edges': gdf_edges,
                        'max_severity': geo_df['edge'].value_counts().max(),
                        'crime_types': geo_df['Incident_Type'].unique().tolist()
                    }
                    st.session_state.temp_filters = {
                        'types': geo_df['Incident_Type'].unique().tolist(),
                        'times': ['Night', 'Morning', 'Afternoon', 'Evening'],
                        'min_severity': 0
                    }
        
        if st.session_state.processed_data:
            st.markdown("---")
            st.subheader("Refine Results")
            
            if 'temp_filters' not in st.session_state:
                st.session_state.temp_filters = st.session_state.filters.copy()
            
            selected_types = st.multiselect(
                "Crime Types",
                options=st.session_state.processed_data['crime_types'],
                default=st.session_state.temp_filters['types'],
                key='temp_types'
            )
            
            selected_times = st.multiselect(
                "Time of Day",
                options=['Night', 'Morning', 'Afternoon', 'Evening'],
                default=st.session_state.temp_filters['times'],
                key='temp_times'
            )
            
            min_severity = st.slider(
                "Minimum Crimes per Street",
                min_value=0,
                max_value=st.session_state.processed_data['max_severity'],
                value=st.session_state.temp_filters['min_severity'],
                key='temp_min_severity'
            )
            
            st.session_state.temp_filters = {
                'types': selected_types,
                'times': selected_times,
                'min_severity': min_severity
            }
            
            if st.button("✅ Apply Filters", use_container_width=True):
                st.session_state.filters = st.session_state.temp_filters.copy()
                st.rerun()

    if st.session_state.processed_data:
        geo_df = st.session_state.processed_data['geo_df']
        filtered_df = geo_df[
            geo_df['Incident_Type'].isin(st.session_state.filters['types']) &
            geo_df['time_of_day'].isin(st.session_state.filters['times'])
        ]
        
        edge_counts = filtered_df['edge'].value_counts().to_dict()
        filtered_edge_counts = {k: v for k, v in edge_counts.items() 
                               if v >= st.session_state.filters['min_severity']}
                
        m = create_crime_map(filtered_df, filtered_edge_counts, 
                            st.session_state.processed_data['gdf_edges'])

        st.subheader("Interactive Crime Map")
        map_key = f"crime_map_{hash(tuple(sorted(filtered_edge_counts.items())))}"
        map_data = st_folium(
            m,
            key=map_key,
            use_container_width=True,
            height=600,
            returned_objects=None,  
            zoom=14,
            center=[37.87, -122.27]  
        )

        if not map_data.get('bounds'):
            default_bounds = [[37.85, -122.30], [37.89, -122.25]]
            south, west = default_bounds[0]
            north, east = default_bounds[1]
        else:
            south_west = map_data['bounds'].get('_southWest', {'lat': 37.85, 'lng': -122.30})
            north_east = map_data['bounds'].get('_northEast', {'lat': 37.89, 'lng': -122.25})
            south = south_west['lat']
            west = south_west['lng']
            north = north_east['lat']
            east = north_east['lng']

            in_bounds_df = filtered_df[
                (filtered_df['latitude'] >= south) &
                (filtered_df['latitude'] <= north) &
                (filtered_df['longitude'] >= west) &
                (filtered_df['longitude'] <= east)
            ]
             
            with st.expander(f"📋 Crimes in Current View ({len(in_bounds_df)} incidents)", expanded=True):
                if not in_bounds_df.empty:
                    st.dataframe(
                        in_bounds_df[['Incident_Type', 'Block_Address', 
                                    'Occurred_Datetime', 'time_of_day']],
                        column_config={
                            'Occurred_Datetime': st.column_config.DatetimeColumn(
                                'Occurred',
                                format='YYYY-MM-DD HH:mm'
                            ),
                            'time_of_day': 'Time of Day',
                            'Block_Address': 'Location'
                        },
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                else:
                    st.info("No crimes found in current map view")
        
        # Statistics
        with st.sidebar:
            st.markdown("---")
            st.metric("Total Crimes", len(filtered_df))
            st.metric("Filtered Street Segments", len(filtered_edge_counts))
            st.markdown("---")
            st.write("Map Legend:")
            st.markdown("- 🟢 Low density")
            st.markdown("- 🟡 Moderate density")
            st.markdown("- 🟠 High density")
            st.markdown("- 🔴 Severe density")
            st.markdown("- 🟣 Critical density")
    else:
        st.info("Select dates and click 'Run Analysis' to start")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Page", 
        ["Crime Dashboard", "Predictive Analytics"],
        key="nav_select"
    )
    
    if page == "Crime Dashboard":
        original_main()
    else:
        predictive_features_page()
        
if __name__ == "__main__":
    main()