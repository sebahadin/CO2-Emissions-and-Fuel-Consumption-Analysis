
# CO₂ Emissions and Fuel Consumption Analysis with Streamlit

## Overview
This Streamlit web application analyzes vehicle CO₂ emissions and fuel consumption data. Through statistical summaries, interactive visualizations, and a predictive model, it helps you understand factors influencing emissions and predict CO₂ output for different vehicle features.

## Features

### 1. Introduction
- **Dataset Overview:** Displays the first few rows of the dataset with interactive filtering options (by columns and row count).
- **Numerical Analysis:** Shows summary statistics (mean, std, etc.), median, mode, range, and missing values for numeric features.

### 2. Data Visualization
- **Scatter Plot:** Visualizes the relationship between fuel consumption (L/100 km) and CO₂ emissions (g/km), color-coded by fuel type.
- **Themed Plot:** Dark background scatter plot for improved contrast.
- **Looker Dashboard Embed:** Integrates a Google Looker Studio dashboard for additional insights.

### 3. CO₂ Emissions Prediction
- **Machine Learning Model:** A linear regression model predicts CO₂ emissions from various vehicle features.
- **Performance Metrics:** Displays mean absolute error (MAE).
- **User Input:** Users can input custom features to get a predicted CO₂ emissions value.
- **Correlation Heatmap:** A heatmap of key numerical features to visualize correlations.

## Technologies Used
- **Python Libraries:** 
  - [Streamlit](https://streamlit.io/) for the interactive web app.
  - [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation.
  - [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization.
  - [scikit-learn](https://scikit-learn.org/) for linear regression and model evaluation.
  - [Pillow](https://pillow.readthedocs.io/) for image processing.
- **Google Looker Studio:** For embedded dashboard.

