import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from PIL import Image










app_mode = st.sidebar.selectbox('Select a page >> ',['01 Introduction','02 Data visualization','03 Prediction'])




df1 = pd.read_csv("co2.csv")

# map each letter to the corresponding fuel type. Z = Regular gasoline, D = Diesel, X = Premium gasoline, E = Ethanol, N = Natural gas

df = df1.replace({"Fuel Type": {"Z": "Regular gasoline", "D": "Diesel", "X": "Premium gasoline", "E": "Ethanol", "N": "Natural gas"}})





if app_mode == '01 Introduction':


    image = Image.open("q.jpg")  # Replace with your image filename

    st.image(image, use_container_width=True)

    # Load the dataset
    df = pd.read_csv("co2.csv")

    # Page Title and Introduction
    st.title("CO₂ Emissions of Vehicles")
    st.header("Understanding Vehicle Emissions")

    st.write("""
    This dataset explores vehicle fuel consumption and CO₂ emissions, aiming to understand 
    how different factors influence emission amount. By analyzing fuel consumption and engine specifications, 
    we can predict CO₂ emissions and identify trends that support environmental policies.
    """)

   # Column descriptions
    st.header("📌 Dataset Column Descriptions")
    st.markdown("""
    **Make**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The company that manufactures the vehicle (e.g., Toyota, Ford).</span>  
    **Model**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The specific name or version of the vehicle under a manufacturer (e.g., Corolla, Mustang).</span>  
    **Vehicle Class**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>A classification based on size, weight, and purpose (e.g., SUV, compact, sedan).</span>  
    **Engine Size (L)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The total volume of all engine cylinders, affecting power and fuel consumption.</span>  
    **Cylinders**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The number of cylinders in the engine, impacting performance and efficiency.</span>  
    **Transmission**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The system that controls power delivery, such as automatic or manual.</span>  
    **Fuel Type**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The kind of fuel the vehicle requires (e.g., gasoline, diesel, electric, hybrid).</span>  
    **Fuel Consumption City (L/100 km)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The amount of fuel consumed per 100 km in urban conditions.</span>  
    **Fuel Consumption Hwy (L/100 km)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The fuel usage per 100 km when driving on highways.</span>  
    **Fuel Consumption Comb (L/100 km)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The average fuel consumption, considering both city and highway driving.</span>  
    **Fuel Consumption Comb (mpg)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The equivalent fuel efficiency measured in miles per gallon.</span>  
    **CO2 Emissions (g/km)**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-size: smaller;'>The amount of carbon dioxide released per kilometer, reflecting environmental impact.</span>  
    """, unsafe_allow_html=True)
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
    
    st.markdown("### CO2 Emissions vs Fuel Consumption")

    # Scatter plot to show the CO2 Emissions against miles per gallon
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set black background
    ax.set_facecolor((0.15, 0.15, 0.2))  # Change the plot background to black
    fig.patch.set_facecolor(((0.15, 0.15, 0.2)))  # Change the figure background to black

    # Scatter plot with Seaborn
    sns.scatterplot(
        data=df, 
        x="Fuel Consumption Comb (L/100 km)", 
        y="CO2 Emissions(g/km)", 
        hue="Fuel Type", 
        ax=ax
    )
   
    # Set title and labels
    ax.set_title("CO2 Emissions vs Fuel Consumption", color="white")
    ax.set_xlabel("Fuel Consumption Comb (L/100 km)", color="white")
    ax.set_ylabel("CO2 Emissions (g/km)", color="white")

    # Change text and ticks to white for visibility
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.title.set_color("white")

    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")  # Change legend text to white
    legend.get_frame().set_facecolor("black")  # Change legend background to black
    legend.get_frame().set_edgecolor("white")  # Change legend border to white
    # Show the updated plot in Streamlit
    st.pyplot(fig)
    # Scatter plot to show the CO2 Emissions against miles per gallon(mpg)
    st.markdown("### CO2 Emissions vs Fuel Consumption (mpg)")
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Set black background
    ax.set_facecolor((0.15, 0.15, 0.2))  # Change the plot background to black
    fig.patch.set_facecolor(((0.15, 0.15, 0.2)))  # Change the figure background to black





    # Scatter plot with Seaborn
    sns.scatterplot(
        data=df, 
        x="Fuel Consumption Comb (mpg)", 
        y="CO2 Emissions(g/km)", 
        hue="Fuel Type", 
        ax=ax
    )
    

    # Set title and labels
    ax.set_title("CO2 Emissions vs Fuel Consumption (mpg)", color="white")
    ax.set_xlabel("Fuel Consumption Comb (mpg)", color="white")
    ax.set_ylabel("CO2 Emissions (g/km)", color="white")
    # Change text and ticks to white for visibility
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.title.set_color("white")

    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("white")

    legend.get_frame().set_facecolor("black")
    legend.get_frame().set_edgecolor("white")

    # Show the plot in Streamlit
    st.pyplot(fig)


    
    #scatter plot to show the miles per gallon against the CO2 Emissions   the column names are Fuel Consumption Comb (L/100 km) and CO2 Emissions(g/km)
    

    st.title("Visualisation in Looker Studio ")

    # Replace with your actual Looker Studio Embed URL
    looker_url = "https://lookerstudio.google.com/embed/reporting/8661ccb3-712d-45a7-b2c4-3a0468114c5a/page/lIl5E"

    # Embed Looker Dashboard
    st.components.v1.iframe(looker_url, width=900, height=600)




    


if app_mode == '04 Prediction':
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

    # Streamlit UI
    st.title("CO2 Emissions Prediction")
    st.write("")
    st.write("### Prediction Performance")
    col1, col2 = st.columns(2)
    st.markdown(
            f'<div style="border: 2px solid gray; padding: 10px; height: 50px; text-align: center; font-weight: bold;">'
            f'Average CO2 Emission:     {df["CO2 Emissions(g/km)"].mean():.2f} g/km'
            '</div>', 
            unsafe_allow_html=True
        )
    st.markdown(
            f'<div style="border: 2px solid gray; padding: 10px; height: 50px; text-align: center; font-weight: bold;">'
            f'Mean Absolute Error:     {mae:.2f} g/km'
            '</div>', 
            unsafe_allow_html=True
        )

    st.write("")
    st.write("### Prediction Result")
    st.sidebar.header("Enter Vehicle Features")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.sidebar.number_input(f"{col}", value=0.0)

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([user_input])
    # Create an empty placeholder for prediction output
    prediction_placeholder = st.empty()

    # Predict button
    if st.sidebar.button("Predict CO2 Emissions"):
        predicted_co2 = linear.predict(input_df)[0]
        
        # Update the placeholder with the prediction while keeping the frame
        prediction_placeholder.markdown(
            f'<div id="prediction-box" style="border: 2px solid gray; padding: 10px; height: 50px; text-align: center; font-weight: bold;">'
            f'{predicted_co2:.2f}'
            '</div>', 
            unsafe_allow_html=True)
    else:
        # Display an empty box initially
        prediction_placeholder.markdown(
            '<div style="border: 2px solid gray; padding: 10px; height: 50px;"></div>',
            unsafe_allow_html=True)
    st.write("###  ")
    st.write("### Dataset Sample")
    st.dataframe(df.head(5))
    print(X.columns)


    # Correlation Heatmap
    numericals = df.drop(columns=["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type","coordinates"])
    st.write("###  ")
    st.write("## Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr_matrix = numericals.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="BrBG", fmt=".2f", linewidths=0.5, ax=ax, annot_kws={"color": "white"})
    ax.figure.set_facecolor((0.15, 0.15, 0.2))
    plt.xticks(color="white")  # Change x-axis (column) labels color
    plt.yticks(color="white")  # Change y-axis (row) labels color
    cbar = ax.collections[0].colorbar  # Get color bar object
    plt.setp(cbar.ax.get_yticklabels(), color="white")  # Change text label color
    st.pyplot(fig)
    