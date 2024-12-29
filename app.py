import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sections import eda,geospatial_maps, predictive_model, alert_system, final_report


# Load your dataset
data = pd.read_csv('data/finaldata.csv')
# Add a section for geospatial maps
# App Title
st.sidebar.title("Seismic Risk Analysis")
# Sidebar Navigation
# st.sidebar.title("Sections 👇")
section = st.sidebar.radio("",[
    "Exploratory Data Analysis",
    "Geospatial Risk Maps", 
    "Earthquake Prediction Model", 
    "Alert System Evaluation", 
    "Final Report & Insights"
])
# Render sections based on user selection
if section == "Exploratory Data Analysis":
    eda.render()
elif section == "Geospatial Risk Maps":
    geospatial_maps.render(data)
elif section == "Earthquake Prediction Model":
    predictive_model.render()
elif section == "Alert System Evaluation":
    alert_system.render()
elif section == "Final Report & Insights":
    final_report.render()

# st.sidebar.info("Created by Vetapalem Vajralu")
