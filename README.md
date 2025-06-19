
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



## 🚀 How to Use This App

You can run this app locally using Python or inside a Docker container.

---

### 🧪 Option 1: Run Locally (Python Environment)

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2. Install dependencies
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

#### 3. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

---

### 🐳 Option 2: Run with Docker

#### 1. Build the Docker image
```bash
docker build -t co2-emission-app .
```

#### 2. Run the container
```bash
docker run -p 8501:8501 co2-emission-app
```

Then open your browser and go to [http://localhost:8501](http://localhost:8501)

---

### 📁 File Structure
```
.
├── streamlit_app.py         # Main Streamlit app
├── requirements.txt         # Dependencies
├── co2.csv                  # Dataset
├── Dockerfile               # Container instructions
├── README.md                # Project documentation
```

---

### 📝 Notes
- The app requires internet access to load the Google Looker Studio dashboard.
