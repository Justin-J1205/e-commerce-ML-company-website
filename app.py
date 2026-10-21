import streamlit as st
import pandas as pd
import numpy as np

# --- ALGORITHM IMPORTS (Running invisibly in the backend) ---
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# --- PAGE CONFIGURATION ---
# initial_sidebar_state="collapsed" hides the sidebar into a menu icon by default.
st.set_page_config(page_title="(Name_here)", page_icon="🛍️", layout="wide", initial_sidebar_state="collapsed")

# --- SESSION STATE FOR ENTRANCE SCREEN ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# ==========================================
# ENTRANCE SCREEN
# ==========================================
if not st.session_state.entered:
    st.title("Welcome to (Name_here)")
    st.subheader("Redefining the modern retail experience.")
    st.write("""
    (Name_here) was designed to provide a seamless, highly personalized shopping journey for every customer. 
    From discovering the perfect product to ensuring completely secure transactions, the platform operates 
    efficiently behind the scenes so the storefront remains flawless.
    
    Click below to explore the store and internal merchant tools.
    """)
    
    # Spacing
    st.write("")
    st.write("")
    
    if st.button("Start Here", type="primary"):
        st.session_state.entered = True
        st.rerun()

# ==========================================
# MAIN APPLICATION (Accessed after clicking "Start Here")
# ==========================================
else:
    # --- HIDDEN MENU (SIDEBAR) ---
    st.sidebar.title("Navigation Menu")
    st.sidebar.write("*(Name_here) Internal & Storefront Features*")
    page = st.sidebar.radio(
        "Select a module:",
        [
            "🏠 Store Overview",
            "📈 Revenue Projections",
            "📝 Customer Feedback",
            "🎯 Product Discovery",
            "🛡️ Security & Checkout",
            "🚪 Retention System",
            "💎 VIP Loyalty Program"
        ]
    )
    
    # Add a button in the menu to return to the entrance screen
    if st.sidebar.button("Log Out / Exit"):
        st.session_state.entered = False
        st.rerun()

    # ==========================================
    # PAGE 1: HOME
    # ==========================================
    if page == "🏠 Store Overview":
        st.title("(Name_here) Dashboard")
        st.write("Welcome to the daily operations overview.")
        st.write("Use the menu in the top left corner to navigate through the storefront features and merchant tools.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Shoppers", "12,450", "+15%")
        col2.metric("Catalog Items", "3,200", "+50")
        col3.metric("Customer Satisfaction", "98.2%", "+2.1%")

    # ==========================================
    # PAGE 2: REVENUE PROJECTIONS
    # ==========================================
    elif page == "📈 Revenue Projections":
        st.title("Financial Projections")
        st.caption("⚙️ Core Algorithm: Linear Regression")
        st.write("This internal tool forecasts expected weekly sales based on the allocated marketing budget.")
        
        # ALGORITHM: LINEAR REGRESSION
        ad_spend = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
        revenue = np.array([250, 450, 680, 900, 1100])
        
        model = LinearRegression()
        model.fit(ad_spend, revenue)
        
        user_spend = st.slider("Select Planned Marketing Budget ($):", min_value=100, max_value=1000, step=50, value=600)
        prediction = model.predict([[user_spend]])
        
        st.success(f"**Based on current trends, the projected revenue is ${prediction[0]:.2f}.**")
        
        chart_data = pd.DataFrame({'Marketing Budget': ad_spend.flatten(), 'Historical Revenue': revenue})
        st.line_chart(chart_data.set_index('Marketing Budget'))

    # ==========================================
    # PAGE 3: CUSTOMER FEEDBACK
    # ==========================================
    elif page == "📝 Customer Feedback":
        st.title("Automated Review Sorting")
        st.caption("⚙️ Core Algorithm: Naive Bayes")
        st.write("To optimize efficiency, the system automatically reads incoming customer reviews and flags them to highlight what needs immediate attention.")

        # ALGORITHM: NAIVE BAYES
        reviews = ["I love this shirt", "Terrible quality broke immediately", "Amazing shoes fit perfectly", "Worst purchase ever", "Highly recommend this"]
        labels = ["Positive", "Negative", "Positive", "Negative", "Positive"]
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(reviews)
        model = MultinomialNB()
        model.fit(X, labels)
        
        test_review = st.text_input("Paste a new customer review here:")
        if test_review:
            test_X = vectorizer.transform([test_review])
            pred = model.predict(test_X)[0]
            
            if pred == "Positive":
                st.success(f"**System Flag: {pred} Feedback. (No action required)**")
            else:
                st.error(f"**System Flag: {pred} Feedback. (Forwarding to Customer Service)**")

    # ==========================================
    # PAGE 4: PRODUCT DISCOVERY
    # ==========================================
    elif page == "🎯 Product Discovery":
        st.title("Storefront: 'You Might Also Like'")
        st.caption("⚙️ Core Algorithm: K-Nearest Neighbors (KNN)")
        st.write("When customers browse an item, the catalog automatically recommends similar products based on price and category.")

        # ALGORITHM: K-NEAREST NEIGHBORS (KNN)
        products = pd.DataFrame({
            'Product Name': ['Running Shoes', 'Sneakers', 'Winter Coat', 'Rain Jacket', 'Socks'],
            'Price': [120, 110, 200, 180, 15],
            'Category_Code': [1, 1, 2, 2, 3]
        })
        st.write("Current Catalog Sample:", products[['Product Name', 'Price']])
        
        features = products[['Price', 'Category_Code']]
        knn = NearestNeighbors(n_neighbors=2) 
        knn.fit(features)
        
        st.write("---")
        st.write("If a shopper is currently looking at 'Running Shoes' ($120)...")
        distances, indices = knn.kneighbors([[120, 1]])
        
        recommended_item = products.iloc[indices[0][1]]['Product Name'] 
        
        st.info(f"**The system will automatically suggest they also buy: {recommended_item}**")

    # ==========================================
    # PAGE 5: SECURITY & CHECKOUT
    # ==========================================
    elif page == "🛡️ Security & Checkout":
        st.title("Payment Gateway Security")
        st.caption("⚙️ Core Algorithm: Support Vector Machine (SVM)")
        st.write("The platform is protected from chargebacks by instantly scanning every purchase for unusual behavior before authorizing the credit card.")

        # ALGORITHM: SUPPORT VECTOR MACHINE (SVM)
        X_transactions = np.array([[50, 10], [120, 15], [30, 5], [2000, 8000], [1500, 9500]])
        Y_fraud = np.array(["Approved", "Approved", "Approved", "Blocked", "Blocked"])
        
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_transactions, Y_fraud)
        
        col1, col2 = st.columns(2)
        amt = col1.number_input("Cart Total ($)", value=100)
        dist = col2.number_input("Distance from Shipping to Billing Address (Miles)", value=20)
        
        if st.button("Process Payment"):
            result = svm_model.predict([[amt, dist]])[0]
            if result == "Blocked":
                st.error("**Transaction Blocked: Suspicious activity detected.**")
            else:
                st.success("**Transaction Approved: Payment successful.**")

    # ==========================================
    # PAGE 6: RETENTION SYSTEM
    # ==========================================
    elif page == "🚪 Retention System":
        st.title("Shopper Retention")
        st.caption("⚙️ Core Algorithm: Decision Tree")
        st.write("Customer login frequency and abandoned cart items are tracked to predict and prevent user churn.")

        # ALGORITHM: DECISION TREE
        X_churn = np.array([[2, 0], [5, 1], [30, 4], [45, 5], [1, 0]])
        Y_churn = np.array(["Active", "Active", "At Risk", "At Risk", "Active"])
        
        tree_model = DecisionTreeClassifier()
        tree_model.fit(X_churn, Y_churn)
        
        days = st.slider("Days since shopper last logged in:", 0, 60, 35)
        cart = st.slider("Items abandoned in shopping cart:", 0, 10, 4)
        
        prediction = tree_model.predict([[days, cart]])[0]
        
        if prediction == "At Risk":
            st.warning("**Account Status: At Risk. The system is automatically emailing a 20% discount code to encourage a return.**")
        else:
            st.success("**Account Status: Active & Healthy.**")

    # ==========================================
    # PAGE 7: VIP LOYALTY PROGRAM
    # ==========================================
    elif page == "💎 VIP Loyalty Program":
        st.title("Loyalty Tier Upgrades")
        st.caption("⚙️ Core Algorithm: Artificial Neural Network (ANN)")
        st.write("The platform evaluates subtle engagement patterns from new users to automatically upgrade highly engaged shoppers to VIP status for free shipping.")

        # ALGORITHM: ARTIFICIAL NEURAL NETWORK (ANN)
        X_behavior = np.array([
            [10, 5, 2, 0],   
            [500, 50, 45, 5], 
            [20, 10, 5, 1],  
            [800, 120, 60, 8] 
        ])
        Y_vip = np.array(["Standard", "VIP Status", "Standard", "VIP Status"])
        
        ann_model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=500, random_state=42)
        ann_model.fit(X_behavior, Y_vip)
        
        st.write("Evaluating a new user's daily metrics...")
        st.write("- **Total Spend:** $100")
        st.write("- **Pages Viewed:** 30")
        st.write("- **Minutes on Site:** 20")
        st.write("- **Emails Opened:** 2")
        
        vip_pred = ann_model.predict([[100, 30, 20, 2]])[0]
        
        if vip_pred == "VIP Status":
            st.success(f"**Evaluation Complete: Account upgraded to {vip_pred}.**")
        else:
            st.info(f"**Evaluation Complete: Account remains {vip_pred}.**")