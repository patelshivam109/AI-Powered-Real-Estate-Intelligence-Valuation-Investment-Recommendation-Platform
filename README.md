# House Price Prediction System

## Project Overview
This project builds a **House Price Prediction System** using machine learning and an interactive dashboard.  
The objective is to predict the selling price of a house based on property attributes such as area, number of bedrooms, bathrooms, grade, location, renovation status, and infrastructure-related factors.

The project includes:
- Data preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training and comparison
- Final model selection
- Dashboard for user input and price prediction

---

## Problem Statement
Accurately estimating house prices is important for buyers, sellers, and real-estate businesses.  
The goal of this project is to develop a regression-based machine learning system that can predict house prices from structured housing data and expose the prediction through a dashboard interface.

---


Dataset Information
Dataset A

Dataset A is the main dataset used for:

preprocessing
feature engineering
EDA
model training and evaluation
Dataset B

Dataset B was provided separately. Its role depends on the project/internship requirement.
In this project, Dataset A was used for the main predictive modelling pipeline.

Features Used for Prediction

The final model uses the following input features:

number of bedrooms
number of bathrooms
living area
lot area
number of floors
waterfront present
number of views
condition of the house
grade of the house
Postal Code
Lattitude
Longitude
living_area_renov
lot_area_renov
Number of schools nearby
Distance from the airport
Property_Age
Renovated
Basement_Percentage
Infrastructure_Score
Feature Engineering

Several additional features were created to improve model performance:

Property_Age = Sale year - Built year
Renovated = 1 if house was renovated, else 0
Basement_Percentage = basement area / total living area
Infrastructure_Score = combined normalized score from:
Number of schools nearby
Distance from airport
Grade of house
Exploratory Data Analysis (EDA)

The EDA focused on:

distribution of target variable Price
skewness of price and use of log transformation
relationship of area, bedrooms, bathrooms with price
impact of grade, condition, views, waterfront
effect of renovation, property age, latitude/longitude
influence of engineered features such as basement percentage and infrastructure score
outlier detection using boxplots and IQR
Key EDA Insights
Price is heavily right-skewed, so log transformation improved target distribution.
Living area is one of the strongest drivers of house price.
Grade of the house has a strong positive relationship with price.
Bathrooms are more informative than bedrooms.
Waterfront houses are significantly more expensive.
Renovated properties tend to have higher prices.
Latitude captures stronger location signal than longitude.
Schools nearby and distance from airport showed weak direct relationship with price.
Models Trained

The following regression models were trained and compared:

Linear Regression
Random Forest Regressor
XGBoost Regressor
CatBoost Regressor
LightGBM Regressor
Gradient Boosting Regressor

Each model was trained on:

Original target (Price)
Log-transformed target (log1p(Price))
Model Evaluation Metrics

The following metrics were used:

MAE – Mean Absolute Error
RMSE – Root Mean Squared Error
R² Score
Model Comparison Results
Model	Target	MAE	RMSE	R²
CatBoost	Log	60713.90	103548.36	0.9239
LightGBM	Log	61886.93	105655.59	0.9208
XGBoost	Log	61453.95	109080.93	0.9156
Gradient Boosting	Log	64571.56	110417.36	0.9135
CatBoost	Original	64388.47	111249.56	0.9122
XGBoost	Original	63562.11	116615.59	0.9035
LightGBM	Original	67601.53	125280.59	0.8886
Gradient Boosting	Original	68149.67	126273.50	0.8868
Random Forest	Original	68158.58	129923.30	0.8802
Random Forest	Log	68766.09	135986.83	0.8688
Linear Regression	Log	106893.83	196359.26	0.7264
Linear Regression	Original	124714.36	197976.29	0.7219
Final Model Selection
Best Model: CatBoost Regressor (Log Target)

Why selected?

Highest R² score
Lowest RMSE
Lowest MAE among the best-performing models
Performed better after handling skewness using log transformation
Dashboard

An interactive dashboard was built to allow users to:

enter house/property details
get predicted house price
view model-based price estimation in a user-friendly interface
Dashboard Inputs

The dashboard accepts the same features used by the final trained model, including:

bedrooms
bathrooms
living area
lot area
floors
views
grade
location
renovation status
infrastructure score
and other house/property details
Installation
1) Clone the repository
git clone <your-repo-url>
cd House-Price-Prediction
1) Create a virtual environment (recommended)
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3) Install dependencies
pip install -r requirements.txt
How to Run
Run EDA notebook

Open:

notebooks/01_eda.ipynb
Run modelling notebook

Open:

notebooks/02_modelling.ipynb
Run dashboard

If the dashboard is built using Streamlit:

streamlit run dashboard/app.py

If it is built differently, replace the command accordingly.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
XGBoost
CatBoost
LightGBM
Jupyter Notebook
Streamlit / dashboard framework used in project
Outputs

The project produces:

trained house price prediction model
saved model artifacts
EDA plots
model evaluation results
interactive dashboard for inference
Future Improvements

Possible improvements:

hyperparameter tuning for top models
SHAP-based explainability for predictions
better location-based features
use of external real-estate or socio-economic data
deployment of dashboard on cloud platform
Author

Shivam Patel

Conclusion

This project successfully builds a machine learning pipeline for house price prediction and compares multiple regression models.
Among all tested models, CatBoost trained on the log-transformed target performed best and was used in the final dashboard system for price prediction.