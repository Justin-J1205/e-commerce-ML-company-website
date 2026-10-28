import streamlit as st
import pandas as pd
import numpy as np
import time
from models import load_and_train_models

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aura Retail Systems | Enterprise AI Portal",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD EXTERNAL CSS ---
def load_css(file_name="style.css"):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# --- SESSION STATE (Portal Authentication / Routing) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- DATA & MODEL INITIALIZATION ---
@st.cache_resource(show_spinner=False)
def init_models():
    return load_and_train_models('Retail_sales.csv')

# ==============================================================================
# LANDING PAGE (Authentication & Company Overview)
# ==============================================================================
if not st.session_state.logged_in:
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Hero Title & Enterprise Header
    st.markdown('''
        <div class="hero-container">
            <div class="hero-title">Aura Retail Systems</div>
            <div class="hero-subtitle">
                Enterprise Data Science & Machine Learning Platform.<br>
                Transforming raw retail inventory into predictive operational intelligence.
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Login & Authentication Card
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">AUTHORIZED ACCESS ONLY</span>', unsafe_allow_html=True)
        st.write("### Employee & Stakeholder Portal")
        st.write("Authenticate to load the machine learning models and internal operational dashboards.")
        
        emp_id = st.text_input("Employee / Student ID", value="EMP-80429")
        dept = st.selectbox(
            "Access Module", 
            ["Executive Overview", "Financial Operations", "Cataloging & Data Entry", "Marketing & E-Commerce", "Purchasing & Procurement", "Logistics & Warehousing", "Supply Chain Risk"]
        )
        
        if st.button("Launch AI Operations Portal"):
            with st.spinner("Initializing Scikit-Learn pipelines & caching models..."):
                init_models()
                time.sleep(1.2)
                st.session_state.logged_in = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# MAIN APPLICATION (Authenticated Portal)
# ==============================================================================
else:
    df, lr_model, nb_model, vectorizer, knn_model, svm_model, dt_model, ann_model = init_models()

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.markdown("## 🌐 Aura Systems")
    st.sidebar.markdown("<small>Enterprise AI Portal | Active Session</small>", unsafe_allow_html=True)
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Departmental Modules",
        [
            "🏢 Executive Dashboard (Overview)",
            "📈 Finance: Sales Forecaster (Linear Regression)",
            "📝 Data Entry: Auto-Categorizer (Naive Bayes)",
            "🛍️ Marketing: Recommendation Engine (KNN)",
            "🛡️ Purchasing: Volume Classifier (SVM)",
            "📦 Logistics: Dead-Stock Router (Decision Tree)",
            "🚨 Supply Chain: Surge Alerts (Neural Network)"
        ]
    )
    
    st.sidebar.divider()
    if st.sidebar.button("Log Out / Reset Portal"):
        st.session_state.logged_in = False
        st.rerun()

    # Page Container with Fade-In Animation
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

    # ==========================================================================
    # 1. EXECUTIVE DASHBOARD (Company Concept & Mission/Vision/Goals)
    # ==========================================================================
    if page == "🏢 Executive Dashboard (Overview)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">EXECUTIVE SUMMARY</span>', unsafe_allow_html=True)
        st.title("Aura Retail Systems — Enterprise Overview")
        st.write(
            "**Aura Retail Systems** is a fictional next-generation beverage and consumer goods retail company. "
            "We operate an extensive multi-regional warehouse network and storefront distribution model, leveraging historical "
            "transactional data (`Retail_sales.csv`) to automate everyday decisions—from inventory stocking to pricing strategies."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Mission, Vision, and Goals Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('''
                <div class="info-card">
                    <h3>🎯 Company Mission</h3>
                    <p>To empower modern retail operations with predictive machine learning intelligence, converting raw transactional inventory data into automated, real-time business actions.</p>
                </div>
            ''', unsafe_allow_html=True)
        with c2:
            st.markdown('''
                <div class="info-card">
                    <h3>👁️ Company Vision</h3>
                    <p>To redefine global retail supply chains by replacing reactive manual management with seamless, mathematically verified AI operational workflows.</p>
                </div>
            ''', unsafe_allow_html=True)
        with c3:
            st.markdown('''
                <div class="info-card">
                    <h3>🚀 Operational Goals</h3>
                    <p>1. Automate SKU categorization.<br>2. Forecast retail revenue accurately.<br>3. Eliminate warehouse dead-stock.<br>4. Prevent seasonal stockouts.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 📊 Active Operational Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items in Dataset", f"{len(df):,}")
        m2.metric("Total Warehouse Units", f"{df['WAREHOUSE SALES'].sum():,.0f}")
        m3.metric("Retail Sales Tracked", f"{df['RETAIL SALES'].sum():,.0f}")
        m4.metric("Active AI Engines", "6 / 6 Models Online")

        st.markdown("---")
        st.markdown("### 📂 Sample Dataset Preview (`Retail_sales.csv`)")
        st.write("Below is a sample of the historical inventory records used to train our backend machine learning algorithms:")
        st.dataframe(df.head(10), use_container_width=True)

    # ==========================================================================
    # 2. LINEAR REGRESSION (Finance)
    # ==========================================================================
    elif page == "📈 Finance: Sales Forecaster (Linear Regression)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">FINANCE & PLANNING</span>', unsafe_allow_html=True)
        st.title("📈 Retail Sales Forecaster")
        st.write("Forecast projected retail sales volume based on scheduled stock transfers and warehouse inventory levels.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Technical & Business Explanation Cards
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>Linear Regression</b> models continuous numerical relationships. In our business, financial analysts must know how much retail revenue a product will generate based on warehouse availability before approving distribution costs.</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_exp2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Features (X):</b> <code>RETAIL TRANSFERS</code>, <code>WAREHOUSE SALES</code><br>
                    <b>Target Variable (y):</b> <code>RETAIL SALES</code><br>
                    <b>Goal Contribution:</b> Eliminates budget over-allocation by accurately forecasting expected sales return.</p>
                </div>
            ''', unsafe_allow_html=True)

        # Interactive Form & Live Inference
        st.markdown("### 🎛️ Interactive Forecast Simulator")
        with st.form("lr_form"):
            fc1, fc2 = st.columns(2)
            transfers_input = fc1.number_input("Scheduled Retail Transfers (Units)", min_value=0, max_value=1000, value=25)
            warehouse_input = fc2.number_input("Current Warehouse Stock (Units)", min_value=0, max_value=10000, value=250)
            
            submit_lr = st.form_submit_button("Run Linear Regression Prediction")
            
            if submit_lr:
                with st.spinner("Computing linear equation..."):
                    time.sleep(0.5)
                    pred_sales = lr_model.predict([[transfers_input, warehouse_input]])[0]
                    st.success(f"📈 **Projected Retail Sales:** **{max(0.0, pred_sales):.2f} units**")

        # Visual Chart
        st.markdown("### 📉 Regression Model Trend Visualization")
        sample_chart_df = df.sample(n=100, random_state=42).sort_values(by='WAREHOUSE SALES')
        sample_chart_df['Predicted Retail Sales'] = lr_model.predict(sample_chart_df[['RETAIL TRANSFERS', 'WAREHOUSE SALES']])
        st.line_chart(sample_chart_df.set_index('WAREHOUSE SALES')[['RETAIL SALES', 'Predicted Retail Sales']])

    # ==========================================================================
    # 3. NAIVE BAYES (Cataloging & Data Entry)
    # ==========================================================================
    elif page == "📝 Data Entry: Auto-Categorizer (Naive Bayes)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">CATALOGING & DATA ENTRY</span>', unsafe_allow_html=True)
        st.title("📝 Automated Product SKU Auto-Categorizer")
        st.write("Automatically classify incoming inventory items into product categories using Natural Language Processing.")
        st.markdown('</div>', unsafe_allow_html=True)

        col_nb1, col_nb2 = st.columns(2)
        with col_nb1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>Naive Bayes</b> applies Bayes' Theorem with feature independence assumptions. It excels at fast text classification by calculating word probabilities across known product categories (e.g., WINE, LIQUOR, BEER).</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_nb2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Feature (X):</b> <code>ITEM DESCRIPTION</code> (Text Count Vectorizer)<br>
                    <b>Target Variable (y):</b> <code>ITEM TYPE</code> (Category Tag)<br>
                    <b>Goal Contribution:</b> Saves thousands of human data-entry hours when processing newly acquired vendor inventory.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 🎛️ Live Text Tagging Tool")
        test_desc = st.text_input("Enter Vendor Product Description", value="CHARDONNAY VINTAGE RESERVA 750ML")
        
        if st.button("Classify Product Category"):
            if test_desc.strip():
                with st.spinner("Calculating word probabilities..."):
                    time.sleep(0.4)
                    vec_text = vectorizer.transform([test_desc.upper()])
                    pred_cat = nb_model.predict(vec_text)[0]
                    probs = nb_model.predict_proba(vec_text)[0]
                    classes = nb_model.classes_

                    st.success(f"🏷️ **Assigned Category:** **{pred_cat}**")
                    
                    # Probability Bar Chart
                    st.markdown("#### 📊 Category Probability Distribution")
                    prob_df = pd.DataFrame({"Category": classes, "Probability": probs}).sort_values(by="Probability", ascending=False)
                    st.bar_chart(prob_df.set_index("Category"))
            else:
                st.warning("Please enter a valid product description.")

    # ==========================================================================
    # 4. K-NEAREST NEIGHBORS (Marketing)
    # ==========================================================================
    elif page == "🛍️ Marketing: Recommendation Engine (KNN)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">MARKETING & E-COMMERCE</span>', unsafe_allow_html=True)
        st.title("🛍️ Storefront Recommendation Engine")
        st.write("Recommend similar items to online shoppers based on Euclidean distance in sales volume and inventory metrics.")
        st.markdown('</div>', unsafe_allow_html=True)

        col_knn1, col_knn2 = st.columns(2)
        with col_knn1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>K-Nearest Neighbors (KNN)</b> measures distance between data points in multi-dimensional space. We use it on our e-commerce site to display 'Customers Also Viewed' items with matching inventory profiles.</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_knn2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Features (X):</b> <code>RETAIL SALES</code>, <code>WAREHOUSE SALES</code><br>
                    <b>Target Output:</b> Top K Nearest Product Descriptions<br>
                    <b>Goal Contribution:</b> Increases cross-selling revenue and customer basket size during checkout.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 🎛️ Product Match Simulator")
        col_k1, col_k2 = st.columns(2)
        target_retail = col_k1.slider("Target Item Retail Sales Volume", 0, 100, 15)
        target_warehouse = col_k2.slider("Target Item Warehouse Stock Level", 0, 500, 100)

        if st.button("Find Nearest Products (K=3 Recommendations)"):
            with st.spinner("Calculating Euclidean distances..."):
                time.sleep(0.5)
                distances, indices = knn_model.kneighbors([[target_retail, target_warehouse]])
                
                st.markdown("#### 🛒 Recommended Products for Website Display:")
                rec_list = []
                for idx in indices[0][1:]: # Skip index 0 as it is self
                    item = df.iloc[idx]
                    rec_list.append({
                        "Product Name": item['ITEM DESCRIPTION'],
                        "Category": item['ITEM TYPE'],
                        "Retail Sales": item['RETAIL SALES'],
                        "Warehouse Stock": item['WAREHOUSE SALES']
                    })
                st.table(pd.DataFrame(rec_list))

    # ==========================================================================
    # 5. SUPPORT VECTOR MACHINE (Purchasing)
    # ==========================================================================
    elif page == "🛡️ Purchasing: Volume Classifier (SVM)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">PURCHASING & PROCUREMENT</span>', unsafe_allow_html=True)
        st.title("🛡️ Supplier Volume & Tier Classifier")
        st.write("Audit inventory items and classify them as High-Volume vs Standard-Volume to optimize supplier contract negotiations.")
        st.markdown('</div>', unsafe_allow_html=True)

        col_svm1, col_svm2 = st.columns(2)
        with col_svm1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>Support Vector Machine (SVM)</b> constructs an optimal hyperplane to separate classes with maximum margin. In procurement, it draws a strict mathematical boundary between standard products and high-margin inventory drivers.</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_svm2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Features (X):</b> <code>WAREHOUSE SALES</code>, <code>RETAIL TRANSFERS</code><br>
                    <b>Target Variable (y):</b> High Volume Item vs Standard Volume Item<br>
                    <b>Goal Contribution:</b> Enables procurement teams to target high-mover SKUs for volume discounts.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 🎛️ Procurement Tier Audit Form")
        with st.form("svm_form"):
            s_col1, s_col2 = st.columns(2)
            w_vol = s_col1.number_input("Warehouse Sales Volume", min_value=0, max_value=5000, value=650)
            t_vol = s_col2.number_input("Scheduled Transfer Units", min_value=0, max_value=500, value=40)
            
            submit_svm = st.form_submit_button("Run SVM Classification Audit")
            
            if submit_svm:
                with st.spinner("Processing SVM decision boundary..."):
                    time.sleep(0.6)
                    res_svm = svm_model.predict([[w_vol, t_vol]])[0]
                    if res_svm == "High Volume Item":
                        st.success(f"💎 **Audit Result:** **{res_svm}** — *Action: Schedule contract renegotiation for bulk rebate.*")
                    else:
                        st.info(f"📦 **Audit Result:** **{res_svm}** — *Action: Apply standard vendor pricing terms.*")

    # ==========================================================================
    # 6. DECISION TREE (Logistics)
    # ==========================================================================
    elif page == "📦 Logistics: Dead-Stock Router (Decision Tree)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">LOGISTICS & WAREHOUSING</span>', unsafe_allow_html=True)
        st.title("📦 Dead-Stock Predictor & Inventory Router")
        st.write("Predict whether an inventory item risks becoming stagnant dead-stock, guiding warehouse logistics directives.")
        st.markdown('</div>', unsafe_allow_html=True)

        col_dt1, col_dt2 = st.columns(2)
        with col_dt1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>Decision Trees</b> split data using flowchart-like binary rules based on feature thresholds. Logistics managers can inspect the clear rule tree to understand exactly why an item is flagged for warehouse retention.</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_dt2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Features (X):</b> <code>MONTH</code>, <code>WAREHOUSE SALES</code><br>
                    <b>Target Variable (y):</b> Active Stock vs Dead Stock<br>
                    <b>Goal Contribution:</b> Maximizes retail floor space efficiency by holding low-velocity items in secondary storage.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 🎛️ Logistics Routing Simulator")
        col_d1, col_d2 = st.columns(2)
        op_month = col_d1.selectbox("Operating Month", list(range(1, 13)), index=7)
        wh_stock = col_d2.slider("Current Warehouse Stock", 0, 1000, 30)

        if st.button("Generate Routing Directive"):
            with st.spinner("Evaluating decision tree nodes..."):
                time.sleep(0.4)
                dt_res = dt_model.predict([[op_month, wh_stock]])[0]
                if "Dead Stock" in dt_res:
                    st.error(f"🚫 **Logistics Directive:** **{dt_res}**")
                else:
                    st.success(f"✅ **Logistics Directive:** **{dt_res}**")

    # ==========================================================================
    # 7. ARTIFICIAL NEURAL NETWORK (Supply Chain)
    # ==========================================================================
    elif page == "🚨 Supply Chain: Surge Alerts (Neural Network)":
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<span class="dept-badge">SUPPLY CHAIN RISK</span>', unsafe_allow_html=True)
        st.title("🚨 Seasonal Demand Surge Predictor")
        st.write("Detect non-linear seasonal demand spikes using a Multi-Layer Perceptron (MLP) Neural Network.")
        st.markdown('</div>', unsafe_allow_html=True)

        col_ann1, col_ann2 = st.columns(2)
        with col_ann1:
            st.markdown('''
                <div class="info-card">
                    <h3>⚙️ Algorithm Purpose & Relevance</h3>
                    <p><b>Artificial Neural Networks (ANN)</b> learn non-linear patterns across interconnected hidden node layers. They capture complex interactions between month seasonality, warehouse levels, and stock transfers that traditional linear models miss.</p>
                </div>
            ''', unsafe_allow_html=True)
        with col_ann2:
            st.markdown('''
                <div class="info-card">
                    <h3>📊 Data Features & Goal Alignment</h3>
                    <p><b>Input Features (X):</b> <code>MONTH</code>, <code>WAREHOUSE SALES</code>, <code>RETAIL TRANSFERS</code><br>
                    <b>Target Variable (y):</b> Demand Surge Predicted vs Normal Demand<br>
                    <b>Goal Contribution:</b> Completely prevents stockouts during major holiday rushes and unexpected demand spikes.</p>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("### 🎛️ Demand Surge Radar")
        with st.form("ann_form"):
            a_c1, a_c2, a_c3 = st.columns(3)
            tar_month = a_c1.number_input("Target Month (1-12)", 1, 12, 11)
            tar_ware = a_c2.number_input("Warehouse Stock", 0, 5000, 1500)
            tar_trans = a_c3.number_input("Scheduled Transfers", 0, 1000, 45)
            
            submit_ann = st.form_submit_button("Run Neural Network Scan")
            
            if submit_ann:
                with st.spinner("Propagating signals through hidden layers..."):
                    time.sleep(0.8)
                    ann_res = ann_model.predict([[tar_month, tar_ware, tar_trans]])[0]
                    if "Surge" in ann_res:
                        st.error(f"⚠️ **CRITICAL ALERT:** **{ann_res}** — *Action: Issue emergency re-order to supplier immediately.*")
                    else:
                        st.success(f"✅ **System Status:** **{ann_res}** — *Action: Inventory levels are sufficient.*")

    st.markdown('</div>', unsafe_allow_html=True)