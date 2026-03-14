# Imports
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Configuration
st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("retail_sales_data.csv")   # dataset from GitHub repo
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Sales Insights", "Model Training & Forecasting"],
)

# Common Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Year'] = df['Date'].dt.year

# 1. Overview Page
if page == "Overview":
    st.title("Retail Sales Forecasting Dashboard 📊")

    st.markdown(
        """
        Welcome to the **Retail Sales Forecasting Dashboard**!

        - 📊 Sales insights and visualizations  
        - 🤖 Machine learning models for prediction  
        - 📈 Interactive forecasting for business decisions
        """
    )

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.write(df.describe())

# 2. Sales Insights
elif page == "Sales Insights":
    st.title("Sales Insights 🛍️")

    st.subheader("Daily Sales Trend")
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()

    fig_daily_sales = px.line(
        daily_sales,
        x="Date",
        y="Sales",
        title="Daily Sales Trend"
    )

    st.plotly_chart(fig_daily_sales, use_container_width=True)

    st.subheader("Sales by Product Category")

    product_sales = df.groupby('Product_Category')['Sales'].sum().reset_index()

    fig_product_sales = px.bar(
        product_sales,
        x='Product_Category',
        y='Sales',
        color='Product_Category',
        text='Sales',
        title='Total Sales by Product Category'
    )

    st.plotly_chart(fig_product_sales, use_container_width=True)

    st.subheader("Sales by Customer Segment")

    segment_sales = df.groupby('Customer_Segment')['Sales'].sum().reset_index()

    fig_segment_sales = px.pie(
        segment_sales,
        names='Customer_Segment',
        values='Sales',
        title='Sales by Customer Segment'
    )

    st.plotly_chart(fig_segment_sales, use_container_width=True)

# 3. Model Training & Forecasting
elif page == "Model Training & Forecasting":

    st.title("Model Training & Forecasting 🤖")

    le = LabelEncoder()

    df['Product_Category'] = le.fit_transform(df['Product_Category'])
    df['Region'] = le.fit_transform(df['Region'])
    df['Customer_Segment'] = le.fit_transform(df['Customer_Segment'])

    features = [
        'Product_Category',
        'Region',
        'Customer_Segment',
        'Units_Sold',
        'Unit_Price',
        'Month',
        'DayOfWeek',
        'Year'
    ]

    X = df[features]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    st.subheader("Random Forest Model")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    st.write(f"MAE: {rf_mae:.2f}")
    st.write(f"RMSE: {rf_rmse:.2f}")

    # XGBoost
    st.subheader("XGBoost Model")

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)

    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    st.write(f"MAE: {xgb_mae:.2f}")
    st.write(f"RMSE: {xgb_rmse:.2f}")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted Sales")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(y_test[:50]))),
        y=y_test[:50],
        mode='lines+markers',
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred_rf[:50]))),
        y=y_pred_rf[:50],
        mode='lines+markers',
        name='RF Predicted'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred_xgb[:50]))),
        y=y_pred_xgb[:50],
        mode='lines+markers',
        name='XGB Predicted'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Future Forecast
    st.subheader("Future Sales Forecast")

    future_data = X.tail(30)

    future_preds_rf = rf_model.predict(future_data)

    fig_future = px.line(
        x=range(len(future_preds_rf)),
        y=future_preds_rf,
        labels={'x': 'Day Index', 'y': 'Forecasted Sales'},
        title="Next 30 Days Forecast"
    )

    st.plotly_chart(fig_future, use_container_width=True)
