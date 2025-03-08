import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from PIL import Image







st.sidebar.header("Select Dataset")




app_mode = st.sidebar.selectbox('Select a page >> ',['01 Introduction','02 Data visualization','04 Prediction'])




df = pd.read_csv("co2.csv")



if app_mode == '01 Introduction':


    image = Image.open("q.jpg")  # Replace with your image filename

    st.image(image, use_container_width=True)

    # Load the dataset
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

    # Numerical Analysis
    st.header("Numerical Analysis")

    # Selecting numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    if not numerical_columns.empty:
        # Summary Statistics
        st.write("### Summary Statistics")
        st.dataframe(df[numerical_columns].describe().T.style.format("{:.2f}"))

        # Additional Metrics
        st.write("### Additional Metrics")
        additional_metrics = pd.DataFrame({
            "Median": df[numerical_columns].median(),
            "Mode": df[numerical_columns].mode().iloc[0],  # Mode might return multiple values; take the first one
            "Range": df[numerical_columns].max() - df[numerical_columns].min(),
            "Missing Values": df[numerical_columns].isnull().sum()
        }).T

        st.dataframe(additional_metrics.style.format("{:.2f}"))
    else:
        st.write("No numerical columns found in the dataset.")




if app_mode == '02 Data visualization':
    st.title("Data Visualization")

    st.title("Google Looker Dashboard in Streamlit")

    # Replace with your actual Looker Studio Embed URL
    looker_url = "https://lookerstudio.google.com/embed/reporting/8661ccb3-712d-45a7-b2c4-3a0468114c5a/page/lIl5E"

    # Embed Looker Dashboard
    st.components.v1.iframe(looker_url, width=900, height=600)




    


if app_mode == '04 Prediction':
    st.title("CO₂ Emissions Prediction")


    # Step 1: Clean data and prepare features
    X = df.drop(columns=["CO2 Emissions(g/km)","Make", "Model", "Vehicle Class", "Transmission", "Fuel Type","coordinates"])
    y = df["CO2 Emissions(g/km)"]

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train Linear Regression model
    linear = LinearRegression()
    linear.fit(X_train, y_train)

    # Step 4: Evaluate model
    predictions = linear.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Correlation Heatmap
    st.write("### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)


    # Streamlit UI
    st.title("CO2 Emissions Prediction")
    st.write("This app predicts CO2 emissions (g/km) based on vehicle attributes using linear regression.")

    st.sidebar.header("Enter Vehicle Features")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Predict button
    if st.sidebar.button("Predict CO2 Emissions"):
        predicted_co2 = linear.predict(input_df)[0]
        st.write("### Prediction Result")
        st.write(f"**Predicted CO2 Emissions (g/km):** {predicted_co2:.2f}")
        st.write(f"Model Mean Absolute Error: {mae:.2f}")

    st.write("### Dataset Sample")
    st.dataframe(df.head())

    