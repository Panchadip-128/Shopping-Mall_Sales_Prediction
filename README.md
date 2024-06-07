# Shopping-Mall_Sales_Prediction
----------------------------------
# Big Mart Sales Prediction

This project aims to predict the sales of products across various stores of a big mart using machine learning algorithms. The dataset used is the Big Mart Sales dataset, which contains sales data for different products across multiple outlets.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
The project involves data preprocessing, exploratory data analysis (EDA), and building a regression model using XGBoost to predict the sales of products. The goal is to understand the factors affecting sales and build a robust predictive model.

## Dataset
The dataset used is the Big Mart Sales dataset, contains the following columns:
- Item_Identifier
- Item_Weight
- Item_Fat_Content
- Item_Visibility
- Item_Type
- Item_MRP
- Outlet_Identifier
- Outlet_Establishment_Year
- Outlet_Size
- Outlet_Location_Type
- Outlet_Type
- Item_Outlet_Sales

## Data Preprocessing
- Loaded the dataset into a Pandas DataFrame.
- Handled missing values in `Item_Weight` and `Outlet_Size` columns.
- Encoded categorical variables using LabelEncoder.
- Split the data into training and testing sets.

## Exploratory Data Analysis (EDA)
- Visualized distributions of numerical features.
- Plotted count plots for categorical features.
- Examined relationships between features and the target variable (`Item_Outlet_Sales`).

## Model Training
- Used the XGBoost regressor to train the model.
- Split the data into training and testing sets.
- Trained the model on the training data.

## Evaluation
- Evaluated the model using the R-squared metric.
- Achieved an R-squared value of approximately `r2_train` on the training data and `r2_test` on the test data.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   
   git clone 
   cd Shopping_Mall_Sales_Prediction.ipynb

2.Install the required packages:

pip install -r requirements.txt

3.Usage
Run the Jupyter notebook or Python script to perform data preprocessing, EDA, and model training:

jupyter notebook Shopping_Mall_Sales_Prediction.ipynb

Make predictions on new data:

# Load the trained model
regressor = XGBRegressor()
regressor.load_model('xgb_model.json')

# Preprocess new data
new_data = pd.read_csv('new_data.csv')
# Apply the same preprocessing steps

# Make predictions
predictions = regressor.predict(new_data)

Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

Thank You
