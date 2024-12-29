import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from datetime import datetime
import plotly.graph_objs as go

def preprocess_data(data):
    """Prepare data for analysis"""
    data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y %H:%M', dayfirst=True)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    data['alert_encoded'] = pd.Categorical(data['alert']).codes
    data['tsunami_encoded'] = pd.Categorical(data['tsunami']).codes
    return data

# def generate_statistics(data):
#     """Generate real-time statistics for the dashboard"""
#     current_time = datetime.now()
#     last_24h = current_time - timedelta(days=1)

#     total_events = len(data)
#     avg_magnitude = data['magnitude'].mean()
#     last_24h_events = len(data[data['date_time'] >= last_24h])
#     most_common_alert = data['alert'].mode().iloc[0]

#     st.metric(label="Total Events", value=f"{total_events:,}")
#     st.metric(label="Average Magnitude", value=f"{avg_magnitude:.2f}")
#     st.metric(label="Events in Last 24h", value=f"{last_24h_events:,}")
#     st.metric(label="Most Common Alert", value=most_common_alert)
import streamlit as st
from datetime import datetime, timedelta

def generate_statistics(data):
    """Generate real-time statistics for the dashboard"""
    current_time = datetime.now()
    last_24h = current_time - timedelta(days=1)

    total_events = len(data)
    avg_magnitude = data['magnitude'].mean()
    # last_24h_events = len(data[data['date_time'] >= last_24h])
    
    # Get the most common alert safely
    alert_mode = data['alert'].mode()
    if not alert_mode.empty:
        most_common_alert = alert_mode.iloc[0]
    else:
        most_common_alert = "#ERROR: Data Insufficient(in that range)"
    
    # Display metrics on Streamlit
    st.metric(label="Total Events", value=f"{total_events:,}")
    st.metric(label="Average Magnitude", value=f"{avg_magnitude:.2f}")
    # st.metric(label="Events in Last 24h", value=f"{last_24h_events:,}")
    st.metric(label="Most Common Alert", value=most_common_alert)


# def create_map(data):
#     """Create an interactive map visualization"""
#     fig = go.Figure(data=go.Scattergeo(
#         lon=data['longitude'],
#         lat=data['latitude'],
#         mode='markers',
#         marker=dict(
#             size=data['magnitude'] * 3,
#             color=data['alert_encoded'],
#             colorscale='Viridis',
#             showscale=True,
#             colorbar=dict(title="Alert Level"),
#         ),
#         text=data.apply(
#             lambda row: f"<b>Magnitude:</b> {row['magnitude']:.1f}<br>"
#                        f"<b>Alert:</b> {row['alert']}<br>"
#                        f"<b>Location:</b> {row['location']}<br>"
#                        f"<b>Depth:</b> {row['depth']}km",
#             axis=1
#         ),
#         hoverinfo='text'
#     ))
#     fig.update_layout(
#         title=dict(text='Geographic Distribution of Seismic Events', x=0.5),
#         geo=dict(
#             showland=True,
#             showcountries=True,
#             showocean=True,
#             landcolor='rgb(243, 243, 243)',
#             oceancolor='rgb(204, 229, 255)',
#             projection_scale=1.2,
#             projection_type='natural earth',
#         ),
#     )
#     return fig
import plotly.graph_objects as go

def create_map(data):
    """Create an interactive 3D globe visualization"""
    
    # Define the alert levels and their numeric encoding
    alert_mapping = {
        'green': 0,     # low alert
        'yellow': 1,    # moderate alert
        'no data': 2,   # no data or missing alert
        'orange': 3,    # high alert
        'red': 4        # very high alert
    }
    
    # Apply the alert encoding to the 'alert' column
    data['alert_encoded'] = data['alert'].map(alert_mapping)
    
    # Set up a color scale with 5 colors (matching the 5 alert levels)
    color_scale = [
        [0, 'green'],    # 0: green
        [0.25, 'yellow'],  # 1: yellow
        [0.5, 'gray'],   # 2: gray for no data
        [0.75, 'orange'], # 3: orange
        [1, 'red']       # 4: red
    ]
    
    fig = go.Figure(data=go.Scattergeo(
        lon=data['longitude'],
        lat=data['latitude'],
        mode='markers',
        marker=dict(
            size=data['magnitude'] * 3,  # Size based on magnitude
            color=data['alert_encoded'],  # Color based on encoded alert level
            colorscale=color_scale,  # Custom color scale
            cmin=0,  # Min value for color scale
            cmax=4,  # Max value for color scale (corresponding to 4 alert levels)
            showscale=True,
            colorbar=dict(title="Alert Level"),
            opacity=0.7  # Adjust the opacity for better visualization
        ),
        text=data.apply(
            lambda row: f"<b>Magnitude:</b> {row['magnitude']:.1f}<br>"
                       f"<b>Alert:</b> {row['alert']}<br>"
                       f"<b>Location:</b> {row['location']}<br>"
                       f"<b>Depth:</b> {row['depth']} km<br>"
                       f"<b>MMI:</b> {row['mmi']}",
            axis=1
        ),
        hoverinfo='text'
    ))

    # Update layout for 3D globe visualization
    fig.update_layout(
        title=dict(text='Geographic Distribution of Seismic Events', x=0.5),
        geo=dict(
            projection_type='orthographic',  # Use orthographic projection for a globe-like effect
            showland=True,
            showcountries=True,
            showocean=True,
            landcolor='rgb(243, 243, 243)',
            oceancolor='rgb(204, 229, 255)',
            showcoastlines=True,
            coastlinecolor="Black",
            projection_scale=1.2
        ),
        scene=dict(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            zaxis=dict(showgrid=False),
            camera=dict(
                eye=dict(x=1, y=1, z=1),  # Initial camera position
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='cube'
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        height=700
    )

    return fig
# Create the map for the entire dataset


def filter_data(data, start_date, end_date, magnitude_range, alert_types, continents):
    """Filters the data based on the input parameters."""
    # Ensure `start_date` and `end_date` are converted to datetime
    start_date = datetime.combine(start_date, datetime.min.time())
    end_date = datetime.combine(end_date, datetime.max.time())

    filtered_data = data[
        (data['date_time'] >= start_date) &
        (data['date_time'] <= end_date) &
        (data['magnitude'] >= magnitude_range[0]) &
        (data['magnitude'] <= magnitude_range[1]) &
        (data['alert'].isin(alert_types)) &
        (data['continent'].isin(continents))
    ]
    return filtered_data

# Main Streamlit App
def render():
    st.title("Alert System Evaluation")

# Load data
    df = pd.read_csv('data/finaldata.csv')
    df = preprocess_data(df)

# Sidebar Controls
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=df['date_time'].min().date())
    end_date = st.sidebar.date_input("End Date", value=df['date_time'].max().date())
    # magnitude_range = st.sidebar.slider("Magnitude Range", float(df['magnitude'].min()), float(df['magnitude'].max()), (float(df['magnitude'].min()), float(df['magnitude'].max())))
    try:
        magnitude_range = st.sidebar.slider(
        "Magnitude Range", 
        float(df['magnitude'].min()), 
        float(df['magnitude'].max()), 
        (float(df['magnitude'].min()), float(df['magnitude'].max()))
    )
    except IndexError:
        st.error("#ERROR: Data Insufficient")
    except ValueError:
        st.error("#ERROR: Data Insufficient")
    alert_types = st.sidebar.multiselect("Alert Types", options=df['alert'].unique(), default=df['alert'].unique())
    continents = st.sidebar.multiselect("Continents", options=df['continent'].unique(), default=df['continent'].unique())

# Filter data
    filtered_data = filter_data(df, start_date, end_date, magnitude_range, alert_types, continents)

# Main Dashboard Content
    # st.header("Alert System Evaluation")
    st.markdown("""
    **Alert System Overview:**  
    The alert system is an important part of earthquake analysis. 
    Different levels of alert are issued based on the magnitude and impact of seismic events. These levels help guide response efforts and inform populations about potential risks.  
    - **Green**: Low alert, minor seismic activity detected.  
    - **Yellow**: Moderate alert, possible regional impact.  
    - **Orange**: High alert, significant regional impact expected.  
    - **Red**: Very high alert, widespread or catastrophic effects expected.
    """)

    # 2. Real-Time Statistics Section
    st.header("Real-Time Statistics")
    st.markdown(""" 
    This section provides key metrics on the seismic activity within the selected filters. Metrics like the total number of events, the average magnitude, and the most frequent alert type give a quick overview of the seismic activity and the performance of the alert system in detecting earthquakes.
    """)

    generate_statistics(filtered_data)

    # 3. Geographic Distribution Section
    st.header("Geographic Distribution")
    st.markdown("""
    This interactive map shows the location of seismic events globally, with markers sized according to their magnitude and colored according to the alert level. The map allows users to explore earthquake activity by region and alert level visually.
    """)

    map_fig = create_map(filtered_data)
    st.plotly_chart(map_fig, use_container_width=True)