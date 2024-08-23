import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pydeck as pdk
from geopy.geocoders import Nominatim

# Define the latitude and longitude ranges for India
india_lat_range = [8, 37]
india_lon_range = [68, 98]

regions = {
    'North': [30, 90, 10, 40],
    'South': [8, 77, 8, 35],
    'East': [21, 90, 8, 24],
    'West': [20, 75, 22, 68]
}

# Function to determine the region based on latitude and longitude
def get_region(latitude, longitude):
    for region, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
            return region
    return 'Unknown'

# Function to preprocess the data
def preprocess_data(data):
    data['Year'] = pd.to_datetime(data['Origin Time'], errors='coerce').dt.year
    data.drop(columns=['Origin Time'], inplace=True, errors='ignore')
    data.dropna(inplace=True)
    return data

# Function to calculate probabilities based on occurrences of nearly similar longitude and latitude
def calculate_probabilities(data):
    threshold = 0.1
    data['Rounded_Longitude'] = data['Longitude'].round(decimals=1)
    data['Rounded_Latitude'] = data['Latitude'].round(decimals=1)
    
    location_counts = data.groupby(['Rounded_Longitude', 'Rounded_Latitude']).size().reset_index(name='Occurrence_Count')
    location_counts['Probability'] = location_counts['Occurrence_Count'] / location_counts['Occurrence_Count'].sum()
    
    data = pd.merge(data, location_counts[['Rounded_Longitude', 'Rounded_Latitude', 'Probability']], 
                    on=['Rounded_Longitude', 'Rounded_Latitude'], how='left')
    
    return data

def train_xgboost_model(data):
    X = data[['Longitude', 'Latitude', 'Magnitude', 'Year', 'Probability']]
    y = data['Probability']
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42)
    
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    st.write(f"Cross-Validation RMSE: {np.sqrt(-cv_results.mean()):.4f}")
    
    model.fit(X, y)
    
    return model

# Function to back-test the model
def back_test_model(data, start_year, end_year):
    train_data = data[data['Year'] <= start_year]
    test_data = data[(data['Year'] > start_year) & (data['Year'] <= end_year)]
    
    X_train = train_data[['Longitude', 'Latitude', 'Magnitude', 'Year', 'Probability']]
    y_train = train_data['Probability']
    X_test = test_data[['Longitude', 'Latitude', 'Magnitude', 'Year', 'Probability']]
    y_test = test_data['Probability']
    
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    st.write(f"Back-Test RMSE for {start_year}-{end_year}: {rmse:.4f}")
    
    return model

# Function to plot earthquakes on map
def plot_earthquakes_map(data):
    st.pydeck_chart(
        pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=20.5937,
                longitude=78.9629,
                zoom=5,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    get_position='[Longitude, Latitude]',
                    get_radius=10000,
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    auto_highlight=True,
                ),
            ],
        )
    )

# Function to get place name based on latitude and longitude
def get_place_name(latitude, longitude):
    geolocator = Nominatim(user_agent="earthquake_prediction_app")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        return location.address if location else "Unknown"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display the Streamlit app
def main_page():
    st.title("Earthquake Prediction in India")
    st.markdown("""
    This website aims to provide insights into earthquake prediction in India.
    """)
    st.markdown("---")
    st.header("Understanding Earthquakes")
    st.markdown("""
    Earthquakes are natural phenomena that occur due to the sudden release of energy in the Earth's crust, causing seismic waves. 
    They can result in widespread devastation and loss of life. Predicting earthquakes can help mitigate their impact and save lives.
    """)
    st.header("Predictions Based on Past Data")
    st.markdown("""
    Our predictions are based on historical earthquake data, including factors such as location, magnitude, and time. 
    By analyzing this data, we aim to identify patterns and trends that can help forecast future earthquake occurrences.
    """)
    
    # Load data
    try:
        data = pd.read_csv('Book1.csv')
    except FileNotFoundError:
        st.error("The data file 'Book1.csv' was not found.")
        return

    # Filter data for India only
    data = data[(data['Latitude'] >= india_lat_range[0]) & (data['Latitude'] <= india_lat_range[1]) &
                (data['Longitude'] >= india_lon_range[0]) & (data['Longitude'] <= india_lon_range[1])]
    
    # Preprocess the data
    data = preprocess_data(data)
    data['Region'] = data.apply(lambda row: get_region(row['Latitude'], row['Longitude']), axis=1)
    
    # Calculate probabilities based on occurrences of nearly similar longitude and latitude
    data = calculate_probabilities(data)
    
    # Train the XGBoost model with K-Fold Cross-Validation
    model = train_xgboost_model(data)
    
    # Back-test the model for the period 2010-2020
    back_test_model(data, start_year=2010, end_year=2020)
    
    # Plot earthquakes on map
    st.subheader('Earthquake-Prone Places Marked')
    top_places = data.groupby(['Longitude', 'Latitude']).size().nlargest(20).index.tolist()
    plot_earthquakes_map(data)

    # Display top 20 most earthquake-prone places in the sidebar
    st.sidebar.subheader('Top 20 Most Earthquake-Prone Places')
    for lon, lat in top_places:
        place_name = get_place_name(lat, lon)
        st.sidebar.write(f"Location: {place_name}, Latitude: {lat}, Longitude: {lon}")

if __name__ == "__main__":
    main_page()
