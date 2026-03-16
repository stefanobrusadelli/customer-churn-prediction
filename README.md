# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn
using transactional retail data. The project demonstrates a complete
data science workflow including **data cleaning, exploratory data
analysis, feature engineering with sliding time windows, walk-forward
validation, and interpretable machine learning models**.

The objective is to identify customers likely to stop purchasing so that
businesses can take **proactive retention actions**.

------------------------------------------------------------------------

# Project Overview

Customer churn prediction is a critical problem for many businesses
because retaining existing customers is often significantly cheaper than
acquiring new ones.

This project builds a predictive modeling pipeline that uses historical
transaction data to estimate the probability that a customer will churn
within a future time window.

The project follows a realistic data science workflow:

1.  Data cleaning and validation
2.  Exploratory data analysis to understand customer behavior
3.  Feature engineering using time-aware sliding windows
4.  Walk-forward model validation to avoid temporal leakage
5.  Model interpretation and business impact analysis

------------------------------------------------------------------------

# Dataset

The project uses the **Online Retail dataset**, which contains
transactional data from a UK-based e-commerce retailer.

The dataset includes:

-   Customer ID
-   Invoice number
-   Product description
-   Quantity purchased
-   Unit price
-   Transaction timestamp
-   Customer country

Each row represents a single product purchased in a transaction.

Dataset source: https://archive.ics.uci.edu/ml/datasets/online+retail

------------------------------------------------------------------------

# Problem Definition

The goal is to predict **customer churn**, defined as a customer who
**does not make a purchase during a future prediction window**.

To simulate a realistic prediction scenario, the dataset is structured
using **observation windows** and **prediction windows**.

Observation Window -- Used to compute behavioral features.\
Prediction Window -- Used to determine whether the customer churned.

This approach ensures the model only uses **historical information
available at prediction time**, preventing data leakage.

------------------------------------------------------------------------

# Project Pipeline

Raw Transaction Data\
↓\
Data Cleaning and Validation\
↓\
Exploratory Data Analysis\
↓\
Feature Engineering (Sliding Time Windows)\
↓\
Walk-Forward Model Training\
↓\
Model Evaluation and Interpretation\
↓\
Business Impact Analysis

------------------------------------------------------------------------

# Repository Structure

    customer-churn-prediction/

    notebooks/
    │
    ├── 01_data_cleaning.ipynb
    ├── 02_eda.ipynb
    ├── 03_feature_engineering.ipynb
    └── 04_modeling.ipynb

    src/
    Feature engineering pipeline and utility functions

    reports/
    Generated figures and analysis outputs

Notebook descriptions:

**01_data_cleaning** - Data audit and schema validation - Handling
missing values and duplicates - Transaction labeling and return
matching - Outlier assessment and dataset validation

**02_eda** - Business metrics and revenue analysis - Customer behavioral
patterns - RFM analysis - Cohort analysis - Statistical testing of churn
signals

**03_feature_engineering** - Sliding window generation - Customer
behavioral feature extraction - Engagement and lifecycle features -
Feature stability diagnostics

**04_modeling** - Walk-forward validation framework - Model training and
hyperparameter tuning - Threshold optimization - SHAP model
interpretation - Business impact analysis

------------------------------------------------------------------------

# Feature Engineering

Features are generated using **sliding observation windows** to capture
customer behavior over time.

Key feature categories include:

**Engagement Features** - Purchase frequency - Average days between
purchases - Number of active purchase periods

**Value Features** - Total spend - Average order value - Customer
lifetime value proxies

**Lifecycle Features** - Customer tenure - Recency since last purchase

**Seasonality Features** - Month and quarter indicators - Temporal
purchase patterns

These features capture both **long-term customer behavior and short-term
activity trends**.

------------------------------------------------------------------------

# Modeling Approach

To avoid temporal data leakage, the project uses **walk-forward
validation**.

Instead of randomly splitting the dataset, the model is trained on past
windows and evaluated on future windows.

Models evaluated include:

-   Random Forest
-   Gradient Boosting
-   XGBoost

Model performance is evaluated using:

-   ROC-AUC
-   Precision
-   Recall
-   F1 score

Additionally, the classification threshold is optimized to balance
recall and precision for churn detection.

------------------------------------------------------------------------

# Model Interpretation

To understand the drivers of churn predictions, **SHAP (SHapley Additive
Explanations)** is used.

This allows us to analyze:

-   Global feature importance
-   Individual prediction explanations
-   Behavioral signals associated with churn risk

Common predictors include:

-   Customer recency
-   Purchase frequency
-   Customer tenure
-   Average order value

------------------------------------------------------------------------

# Business Impact

Predicting churn allows businesses to implement **targeted retention
strategies**, such as:

-   Personalized promotions
-   Customer engagement campaigns
-   Loyalty incentives

By identifying high-risk customers early, companies can reduce revenue
loss and improve long-term customer value.

------------------------------------------------------------------------

# Future Work

Several extensions could further improve the modeling framework.

Approaches such as recurrent neural networks or
transformer architectures may capture temporal dependencies that
aggregated features cannot fully represent.

Finally, the pipeline could be extended to support **production
deployment**, including automated feature generation, scheduled model
retraining, and monitoring of prediction drift over time.

------------------------------------------------------------------------

# Requirements

Main Python libraries used in this project:

-   pandas
-   numpy
-   scikit-learn
-   xgboost
-   matplotlib
-   seaborn
-   shap
-   tqdm

Install dependencies using:

    pip install -r requirements.txt

------------------------------------------------------------------------

# Author

Stefano Brusadelli

GitHub: https://github.com/stefanobrusadelli

------------------------------------------------------------------------

# License

This project is provided for educational and portfolio purposes.
