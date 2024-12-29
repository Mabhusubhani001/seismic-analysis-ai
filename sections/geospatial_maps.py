import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def create_earthquake_map(df):
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='markers',
        marker=dict(
            size=df['magnitude'] * 3,
            color=df['depth'],
            colorscale='Viridis',
            colorbar=dict(title='Depth (km)'),
            sizemin=4,
            sizeref=1
        ),
        hovertemplate="<b>%{text}</b><br>" +
                      "Magnitude: %{marker.size:.1f}<br>" +
                      "Depth: %{marker.color:.1f} km<br>" +
                      "MMI: %{customdata[0]}<br>" +  # Adding MMI information to hover
                      "<extra></extra>",
        text=df['location'],
        customdata=df[['mmi']]  # Pass the MMI data to customdata
    ))

    fig.update_layout(
        height=800,
        width=1000,
        geo=dict(
            scope='world',
            projection_type='orthographic',
            showland=True,
            showcountries=True,
            showocean=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            oceancolor='rgb(230, 230, 250)',
            projection=dict(
                rotation=dict(lon=10, lat=20, roll=0),
                scale=1.2
            ),
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgb(150, 150, 150)',
            coastlinewidth=1
        )
    )

    return fig

def render(df):
    st.header("Geospatial Earthquake Map")
    st.subheader("Global Earthquake Distribution: Magnitude, Depth, and MMI Analysis")
    fig = create_earthquake_map(df)
    st.plotly_chart(fig, use_container_width=True)  # Streamlit-friendly display
    st.write("This map visualizes the global distribution of earthquakes based on their location, magnitude, depth, and MMI (Modified Mercalli Intensity).")
    st.write("Each marker represents an earthquake, with the size of the marker corresponding to its magnitude, and the color reflecting the depth of the earthquake.")    
    st.write("Additionally, the **hover feature** displays details about the earthquake, including:")     
    st.write("**Magnitude**: The size of the earthquake, indicating its energy release.")
    st.write("**Depth**: The depth at which the earthquake occurred, which can influence the impact on the surface.")
    st.write("**MMI**: The Modified Mercalli Intensity, showing the perceived intensity of shaking at specific locations, providing insights into the earthquake's effects on people, buildings, and the environment.")

    st.write("**Insights**:")
    st.write("* Magnitude and Depth: The size of the markers gives a quick overview of earthquake strength, while the color gradient shows the depth. Shallow earthquakes (with smaller depths) tend to cause more surface-level damage, whereas deeper earthquakes might be less felt but still pose a risk.")
    st.write("* Global Patterns: By examining the map, one can identify seismic zones with higher concentrations of earthquakes, often along tectonic plate boundaries.")
    st.write("* Tsunami Risk: The combination of magnitude and depth can also help assess the potential for tsunami generation, as deep and powerful undersea earthquakes can trigger tsunamis.")
    st.write("* Impact: The MMI displayed in the hover feature shows how different regions experience shaking, indicating the relative severity and potential damage in those areas.")

    st.write(" This map serves as a tool for analyzing global earthquake trends and can be used to study the correlation between earthquake parameters and their impacts on the surrounding environment.")