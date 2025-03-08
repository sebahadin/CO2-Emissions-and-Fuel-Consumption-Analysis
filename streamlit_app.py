import streamlit as st
import pandas as pd
import math
from pathlib import Path


df = pd.read_csv("co2.csv")

# Page Title and Introduction
st.title("CO₂ Emissions and Fuel Consumption Analysis")
st.header("Understanding Vehicle Emissions")

st.write("""
This dataset explores vehicle fuel consumption and CO₂ emissions, aiming to understand 
how different factors influence pollution levels. By analyzing fuel consumption and engine specifications, 
we can predict CO₂ emissions and identify trends that support environmental policies.
""")

# Filtering Options
st.subheader("Filter Data")

# Column Selection
columns = st.multiselect("Select Columns to Display", df.columns, default=df.columns.tolist())
filtered_df = df[columns]

# Row Filtering
num_rows = st.slider("Select number of rows to display", min_value=1, max_value=len(df), value=10)
filtered_df = filtered_df.head(num_rows)

# Display Filtered Data
st.subheader("Dataset Overview")
st.dataframe(filtered_df)

# Dataset Description
st.header("Numerical Analysis")

# Selecting numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns

if not numerical_columns.empty:
    st.write("### Summary Statistics")
    st.dataframe(df[numerical_columns].describe().T.style.format("{:.2f}"))

    st.write("### Additional Metrics")
    stats_df = pd.DataFrame({
        "Mean": df[numerical_columns].mean(),
        "Median": df[numerical_columns].median(),
        "Standard Deviation": df[numerical_columns].std()
    }).T

    st.dataframe(stats_df.style.format("{:.2f}"))
else:
    st.write("No numerical columns found in the dataset.")




st.title("Google Looker Dashboard in Streamlit")

# Replace with your actual Looker Studio Embed URL
looker_url = "https://lookerstudio.google.com/embed/reporting/8661ccb3-712d-45a7-b2c4-3a0468114c5a/page/lIl5E"

# Embed Looker Dashboard
st.components.v1.iframe(looker_url, width=900, height=600)