import streamlit as st
# from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.backends.backend_agg import RendererAgg
from streamlit_autorefresh import st_autorefresh
from PIL import Image

ndvi_image = Image.open('ndvi.png')
height_image = Image.open('plant_height.png')

# Set the interval to 10000 milliseconds (10 seconds), and key to refresh
st_autorefresh(interval=30000, key="refresh")

def load_data():
    with open('plant_data_log.json', 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Function to refresh data
def refresh_data():
    st.cache_data.clear()
    return load_data()

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

df = load_data()

st.title('Plant Growth and Health Dashboard')

# Custom Threshold Settings
ndvi_threshold = st.sidebar.number_input('Set Custom NDVI Threshold', min_value=100.0, max_value=250.0, value=150.0)
height_threshold = st.sidebar.number_input('Set Custom Height Threshold (mm)', min_value=10.0, max_value=100.0, value=70.0)

# Add a button to refresh data
if st.button('Refresh Data'):
    df = refresh_data()

st.image(ndvi_image, caption='NDVI Color Mapped Image')

# NDVI Warning
latest_ndvi = df['avg_ndvi'].iloc[-1]
if latest_ndvi < ndvi_threshold:
    st.warning(f"Warning: The plant is showing signs of being unhealthy with an NDVI reading of {latest_ndvi}.")

st.image(height_image, caption='Edge detection height measrued Image')

# Harvest Reminder
latest_height = df['plant_height_mm'].iloc[-1]
if latest_height >= height_threshold:
    st.success(f"Reminder: The plant has reached a height of {latest_height}mm and might be ready for harvest.")

# Convert timestamps to dates for the slider
#df['date'] = df['timestamp'].dt.date

# Check and ensure min and max dates are different
#min_date, max_date = df['date'].min(), df['date'].max()
#if min_date == max_date:
#st.error("Minimum and maximum dates are the same. Please check your data.")
#else:
#    start_date, end_date = st.slider(
#        "Select Date Range",
#        min_value=min_date,
#        max_value=max_date,
#        value=(min_date, max_date)
#    )
#    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

_lock = RendererAgg.lock

with _lock:
    st.subheader('Plant Height Over Time')
    plt.figure(figsize=(10, 4))
    plt.plot(df['timestamp'].values, df['plant_height_mm'].values)  # Convert to NumPy arra
    #plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('Height (mm)')
    plt.ylim(0, 100)
    st.pyplot(plt)

    # Average NDVI Over Time
    st.subheader('Average NDVI Over Time')
    plt.figure(figsize=(10, 4))
    plt.plot(df['timestamp'].values, df['avg_ndvi'].values)  # Convert to NumPy array
    #plt.ylim(100, 250)
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('Average NDVI')
    plt.ylim(100, 250)
    st.pyplot(plt)

    # Soil Moisture Status Over Time
    st.subheader('Soil Moisture Status Over Time')
    plt.figure(figsize=(10, 4))
    plt.plot(df['timestamp'].values, df['soil_moisture_is_dry'].values)  # Convert to NumPy arra
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('Soil Moisture (Dry: 1, Wet: 0)')
    plt.ylim(-1, 2)
    st.pyplot(plt)
# Download button
csv_data = convert_df_to_csv(df)
st.download_button(
    label="Download Data as CSV",
    data=csv_data,
    file_name='plant_data.csv',
    mime='text/csv'
)
