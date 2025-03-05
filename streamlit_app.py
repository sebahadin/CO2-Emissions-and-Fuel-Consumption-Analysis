import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st
import pandas as pd

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("co2.csv")  # Ensure x.csv is in the same directory or provide full path

df = load_data()

# Business Case Presentation
st.title("Business Case & Data Presentation")
st.header("Business Case Overview")
st.write("""
This section introduces the dataset, explaining its purpose and relevance to the business case.
It highlights key attributes and how the data can be used for decision-making.
""")

# Dataset Overview
st.header("Dataset Overview")
st.write("### Sample Data:")
st.dataframe(df.head())  # Display first few rows

# Filtering Options
st.sidebar.header("Filter Options")

# Column Selection
columns = st.sidebar.multiselect("Select Columns to Display", df.columns, default=df.columns.tolist())
filtered_df = df[columns]

# Row Filtering
st.sidebar.subheader("Row Filtering")
num_rows = st.sidebar.slider("Select number of rows to display", min_value=1, max_value=len(df), value=5)
filtered_df = filtered_df.head(num_rows)

# Display Filtered Data
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

# Dataset Description
st.header("Dataset Description")
st.write("### Summary Statistics (Numerical Columns)")
st.write(df.describe())  # Shows count, mean, std, min, 25%, 50% (median), 75%, max

# Additional Numerical Analysis
st.write("### Additional Numerical Description")
numerical_columns = df.select_dtypes(include=['number']).columns

if not numerical_columns.empty:
    st.write(f"Mean values:\n{df[numerical_columns].mean()}")
    st.write(f"Median values:\n{df[numerical_columns].median()}")
    st.write(f"Standard deviation:\n{df[numerical_columns].std()}")
else:
    st.write("No numerical columns found in the dataset.")

st.write("### Data Information")
st.text(df.info())  # Prints dataframe structure

st.write("""
### Notes:
- Use the sidebar filters to adjust the dataset view.
- The statistics section provides insights into numerical values in the dataset.
""")


