import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

@st.cache_resource(show_spinner=False)
def load_and_train_models(file_path='Retail_sales.csv'):
    # Load dataset (supports either Retail_sales.csv or export.csv)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        try:
            df = pd.read_csv('export.csv')
        except FileNotFoundError:
            st.error(f"⚠️ Neither '{file_path}' nor 'export.csv' was found in the project folder.")
            st.stop()

    # Data Cleaning & Transformation
    df = df.dropna(subset=['ITEM DESCRIPTION', 'ITEM TYPE', 'RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES'])
    for col in ['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna()

    # Sample dataset for optimal responsiveness (10,000 rows max)
    df_sampled = df.sample(n=min(10000, len(df)), random_state=42).copy()

    # -------------------------------------------------------------
    # 1. Linear Regression (Sales Forecasting)
    # Features: RETAIL TRANSFERS, WAREHOUSE SALES -> Target: RETAIL SALES
    # -------------------------------------------------------------
    lr = LinearRegression()
    X_lr = df_sampled[['RETAIL TRANSFERS', 'WAREHOUSE SALES']]
    y_lr = df_sampled['RETAIL SALES']
    lr.fit(X_lr, y_lr)

    # -------------------------------------------------------------
    # 2. Naive Bayes (Automated Product Categorization)
    # Feature: ITEM DESCRIPTION (Text Vectorizer) -> Target: ITEM TYPE
    # -------------------------------------------------------------
    cv = CountVectorizer(max_features=1000)
    X_text = cv.fit_transform(df_sampled['ITEM DESCRIPTION'])
    nb = MultinomialNB()
    nb.fit(X_text, df_sampled['ITEM TYPE'])

    # -------------------------------------------------------------
    # 3. K-Nearest Neighbors (Product Recommendation Engine)
    # Features: RETAIL SALES, WAREHOUSE SALES
    # -------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=4)
    knn.fit(df_sampled[['RETAIL SALES', 'WAREHOUSE SALES']])

    # -------------------------------------------------------------
    # 4. Support Vector Machine (Premium Volume Classifier)
    # Features: WAREHOUSE SALES, RETAIL TRANSFERS -> Target: High vs Standard Volume
    # -------------------------------------------------------------
    df_svm = df_sampled.sample(n=min(1500, len(df_sampled)), random_state=42).copy()
    median_sales = df_svm['RETAIL SALES'].median()
    svm_y = np.where(df_svm['RETAIL SALES'] > median_sales, "High Volume Item", "Standard Volume Item")
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(df_svm[['WAREHOUSE SALES', 'RETAIL TRANSFERS']], svm_y)

    # -------------------------------------------------------------
    # 5. Decision Tree (Dead-Stock Inventory Routing)
    # Features: MONTH, WAREHOUSE SALES -> Target: Dead Stock vs Active Stock
    # -------------------------------------------------------------
    dt_y = np.where(df_sampled['RETAIL SALES'] == 0, "Dead Stock (Keep in Warehouse)", "Active Stock (Send to Store Floor)")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(df_sampled[['MONTH', 'WAREHOUSE SALES']], dt_y)

    # -------------------------------------------------------------
    # 6. Artificial Neural Network (Demand Surge Predictor)
    # Features: MONTH, WAREHOUSE SALES, RETAIL TRANSFERS -> Target: Demand Surge Alert
    # -------------------------------------------------------------
    mean_transfers = df_sampled['RETAIL TRANSFERS'].mean()
    ann_y = np.where(df_sampled['RETAIL TRANSFERS'] > (mean_transfers + 5), "High Demand Surge Predicted", "Normal Demand Pattern")
    ann = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42)
    ann.fit(df_sampled[['MONTH', 'WAREHOUSE SALES', 'RETAIL TRANSFERS']], ann_y)

    return df_sampled, lr, nb, cv, knn, svm, dt, ann