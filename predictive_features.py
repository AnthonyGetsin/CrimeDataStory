import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import osmnx as ox
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from streamlit_folium import st_folium
from scipy.stats import gaussian_kde
import folium

def predictive_features_page():
    st.title("🔮 Exploring Predictive Features")
    st.markdown("Here we explore potential features that could be used to predict crime.")

    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.info("Please run the main Crime Dashboard analysis first to load the data.")
        return

    geo_df = st.session_state.processed_data['geo_df'].copy()


    st.subheader("Time-Based Features")
    st.markdown("Analyzing crime trends over time can reveal important patterns.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Hourly Distribution")
        geo_df['hour'] = geo_df['Occurred_Datetime'].dt.hour
        hourly_counts = geo_df['hour'].value_counts().sort_index()
        fig_hourly, ax_hourly = plt.subplots()
        hourly_counts.plot(kind='bar', ax=ax_hourly)
        ax_hourly.set_xlabel("Hour of Day")
        ax_hourly.set_ylabel("Number of Crimes")
        st.pyplot(fig_hourly)

    with col2:
        st.markdown("#### Day of Week Distribution")
        geo_df['day_of_week'] = geo_df['Occurred_Datetime'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = geo_df['day_of_week'].value_counts().reindex(day_order)
        fig_day, ax_day = plt.subplots()
        day_counts.plot(kind='bar', ax=ax_day)
        ax_day.set_xlabel("Day of Week")
        ax_day.set_ylabel("Number of Crimes")
        st.pyplot(fig_day)

    st.subheader("Crime Type History")
    st.markdown("The history of crime types at a specific location might be predictive.")

    if not geo_df.empty:
        st.markdown("#### Most Frequent Crime Types by Location (Top 5)")
        def top_crime_types(series):
            return series.value_counts().head(5).index.tolist()

        crime_type_by_location = geo_df.groupby('Block_Address')['Incident_Type'].apply(top_crime_types)
        st.dataframe(crime_type_by_location.head())
    else:
        st.info("No crime data to analyze crime type history.")

    st.subheader("Spatial Clustering for Identifying Crime Hotspots")
    st.markdown("Clustering algorithms can help identify areas with a high concentration of crime incidents.")

    if not geo_df.empty:
        st.markdown("#### DBSCAN Clustering")
        st.markdown("DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed together, marking as outliers points that lie alone in low-density regions.")

        coords = geo_df[['latitude', 'longitude']].values
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(coords)

        eps = st.slider("DBSCAN Epsilon (Neighborhood Radius)", 0.001, 0.05, 0.01, 0.001)
        min_samples = st.slider("DBSCAN Min Samples (Points in Neighborhood)", 2, 20, 5, 1)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_coords)
        geo_df['cluster'] = clusters

        n_clusters = len(np.unique(clusters)) - 1
        st.write(f"Estimated number of clusters: {n_clusters}")

        if n_clusters > 0:
            fig_clusters, ax_clusters = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=geo_df, palette='viridis', s=20, ax=ax_clusters)
            ax_clusters.set_title("DBSCAN Identified Crime Clusters")
            st.pyplot(fig_clusters)

            if len(np.unique(clusters)) > 2 and len(clusters) > len(np.unique(clusters)):
                silhouette_avg = silhouette_score(scaled_coords, clusters)
                st.write(f"Silhouette Score: {silhouette_avg:.2f} (higher is better)")
            else:
                st.warning("Silhouette Score not calculated as the number of clusters is insufficient.")
        else:
            st.info("No clusters found with the current parameters.")
    else:
        st.info("No crime data available for clustering.")

    st.subheader("Kernel Density Estimation for Crime Intensity")
    if not geo_df.empty:
        st.markdown("#### 2D KDE of Crime Locations")
        try:
            x = geo_df['longitude']
            y = geo_df['latitude']

            kde = gaussian_kde([x, y])

            xmin, xmax = x.min() - 0.01, x.max() + 0.01
            ymin, ymax = y.min() - 0.01, y.max() + 0.01
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kde(positions).T, xx.shape)

            fig_kde, ax_kde = plt.subplots(figsize=(10, 8))
            contour = ax_kde.contourf(xx, yy, f, cmap='viridis')
            ax_kde.scatter(x, y, s=5, color='k', alpha=0.05)
            ax_kde.set_title("Kernel Density Estimation of Crime Locations")
            fig_kde.colorbar(contour, ax=ax_kde, label='Crime Density')
            st.pyplot(fig_kde)
        except Exception as e:
            st.error(f"Error performing Kernel Density Estimation: {e}")
    else:
        st.info("No crime data for Kernel Density Estimation.")


if __name__ == "__main__":
    st.warning("This script is intended to be run as part of the main Streamlit application.")