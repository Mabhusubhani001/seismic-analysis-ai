import streamlit as st
import pandas as pd
import plotly.express as px

# Load your consolidated dataset
data = pd.read_csv('data/finaldata.csv')

# Convert date_time to datetime format and extract the year
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce')
data['year'] = data['date_time'].dt.year

def render():
    st.header("Exploratory Data Analysis (EDA)")
    st.text("EDA provides insights that can guide feature engineering, model development, and decision-making. In the context of earthquake data, EDA will allow us to analyze trends, identify key factors influencing seismic activity, and detect anomalies or patterns that are crucial for risk assessment and prediction.")

    # 1. Earthquake Magnitude Distribution
    st.subheader("1) Earthquake Magnitude Distribution")
    fig = px.histogram(data, x='magnitude', nbins=20, marginal='box', color_discrete_sequence=['blue'], title='Distribution of Earthquake Magnitudes')
    fig.update_layout(xaxis_title="Magnitude", yaxis_title="Frequency")
    st.plotly_chart(fig)
    st.write("This visualization displays the distribution of earthquake magnitudes. It helps us understand the frequency of different magnitude ranges and identify any skewness or outliers in the dataset.")

    # 2. Number of Earthquakes by Year
    st.subheader("2) Number of Earthquakes by Year")
    earthquakes_by_year = data['year'].value_counts().sort_index()
    year_df = pd.DataFrame({"Year": earthquakes_by_year.index, "Count": earthquakes_by_year.values})
    fig = px.line(year_df, x="Year", y="Count", markers=True, title="Number of Earthquakes by Year (1995-2023)")
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Earthquakes")
    st.plotly_chart(fig)
    st.write("This chart shows the yearly count of earthquakes from 1995 to 2023. It helps identify trends over time, such as increasing or decreasing seismic activity in recent years.")

    # 3. Depth vs Magnitude
    # Map 'alert' values to descriptive labels, including "No Data"
    alert_labels = {
        'green': 'Green (Low Alert)',
        'yellow': 'Yellow (Moderate Alert)',
        'orange': 'Orange (High Alert)',
        'red': 'Red (Severe Alert)',
        None: 'No Data',  # For missing values
        '': 'No Data'     # For empty strings
    }

# Create a new column with descriptive alert labels
    data['alert_label'] = data['alert'].map(alert_labels).fillna('No Data')  # Default to 'No Data' for unmapped values

# Plot with updated labels
    st.subheader("3) Depth vs Magnitude")
    fig = px.scatter(data, x='depth', y='magnitude', color='alert_label', 
                    title='Depth vs Magnitude (Colored by Alert Levels)',
                    labels={"depth": "Depth (km)", "magnitude": "Magnitude", "alert_label": "Alert Level"},
                    color_discrete_map={
                        'Green (Low Alert)': 'green',
                        'Yellow (Moderate Alert)': 'yellow',
                        'Orange (High Alert)': 'orange',
                        'Red (Severe Alert)': 'red',
                        'No Data': 'gray'  # Gray for No Data
                    })
    st.plotly_chart(fig)
    st.write("A scatter plot of earthquake depth versus magnitude, color-coded by alert levels. It highlights the relationship between depth and magnitude while indicating the alert severity for various events.")

    # 4. Tsunami Indicator vs Magnitude
    st.subheader("4) Tsunami Indicator vs Magnitude")
    fig = px.box(data, x='tsunami', y='magnitude', color='tsunami', title='Magnitude Distribution for Tsunami vs Non-Tsunami Earthquakes',
                 labels={"tsunami": "Tsunami (0: No, 1: Yes)", "magnitude": "Magnitude"},
                 color_discrete_sequence=px.colors.diverging.Tealrose)
    st.plotly_chart(fig)
    st.write("This box plot compares the magnitude distributions for tsunami-associated and non-tsunami earthquakes. It reveals whether tsunami-inducing earthquakes tend to have higher magnitudes.")

    # 5. Earthquake Counts by Continent
    st.subheader("5) Earthquake Counts by Continent")
    continent_counts = data['continent'].value_counts().reset_index()
    continent_counts.columns = ['Continent', 'Count']
    st.write("Unique Continents List:", continent_counts['Continent'].unique())  # Debugging output
    fig = px.bar(continent_counts, x='Continent', y='Count', color='Continent', title='Earthquake Counts by Continent',
    color_discrete_sequence=px.colors.qualitative.Set1)  # Simpler palette
    st.plotly_chart(fig)
    st.write("This bar chart shows the number of earthquakes recorded on each continent. It provides an overview of regional seismic activity, highlighting which continents are more prone to earthquakes.")

    # 6. Earthquake Counts by Country
    st.subheader("6) Earthquake Counts by Country")
    country_counts = data['country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    st.write("Unique Countries List:", country_counts['Country'].unique())  # Debugging output
    fig = px.bar(country_counts, x='Country', y='Count', color='Country', title='Earthquake Counts by Country',
    color_discrete_sequence=px.colors.qualitative.Set1)  # Simpler palette
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.write("A bar chart that breaks down earthquake counts by country. It helps identify the countries most affected by seismic activity and provides a detailed geographical view of earthquake distribution.")


    # 7. CDI vs Alert Level
    data['alert_label'] = data['alert'].map(alert_labels).fillna('No Data')  # Default to 'No Data' for unmapped values

# Plot with updated labels and colors
    st.subheader("7) CDI vs Alert Level")
    fig = px.box(data, x='alert_label', y='cdi', color='alert_label', 
                title='CDI vs Alert Level',
                labels={"alert_label": "Alert Level", "cdi": "CDI"},
                color_discrete_map={
                    'Green (Low Alert)': 'green',
                    'Yellow (Moderate Alert)': 'yellow',
                    'Orange (High Alert)': 'orange',
                    'Red (Severe Alert)': 'red',
                    'No Data': 'gray'  # Gray for No Data
                })
    st.plotly_chart(fig)
    st.write("This box plot examines the relationship between the Community Internet Intensity (CDI) and alert levels. It shows how CDI varies across different alert categories, which can indicate the severity of the perceived impact.")

    # # 8. Geospatial Insights
    # st.subheader("Geospatial Insights")
    # fig = px.scatter_geo(data, lat='latitude', lon='longitude', color='magnitude', size='depth',
    #                      title='Geospatial Distribution of Earthquakes',
    #                      color_continuous_scale=px.colors.sequential.Viridis,
    #                      labels={"longitude": "Longitude", "latitude": "Latitude"})
    # st.plotly_chart(fig)

    # 9. Time Trends of Earthquakes
    st.subheader("8) Time Trends of Earthquakes")
    fig = px.histogram(data, x='year', color_discrete_sequence=['orange'], title="Earthquakes Per Year",
                       labels={"year": "Year", "count": "Number of Earthquakes"})
    fig.update_layout(xaxis_title="Year", yaxis_title="Count")
    st.plotly_chart(fig)
    st.write("A histogram showing the number of earthquakes per year. It helps to spot long-term temporal patterns and assess if there is an increase or decrease in seismic activity over the years.")
