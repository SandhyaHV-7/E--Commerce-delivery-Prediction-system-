import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Set page configuration
st.set_page_config(page_title="Delivery Prediction", layout="wide")

# Header banner image
st.image("banner_image.jpeg", use_container_width=True)

# Informative content below banner
st.markdown("""
### 📦 About the E-Commerce Delivery Prediction System

With the rapid growth of e-commerce, ensuring timely delivery has become a top priority for logistics and supply chain companies. Late deliveries can lead to customer dissatisfaction, refund costs, and lost reputation.

This intelligent system uses **Machine Learning and AI** to predict whether a package will be delivered **on time** based on factors like:

- Warehouse location  
- Delivery distance  
- Traffic conditions  
- Product type  
- Shipping mode  
- Customer location  

By analyzing shipment details before dispatch, companies can proactively manage deliveries and reduce delays.
""")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: linear-gradient(to bottom right, #ffffff, #e6f7ff);
        padding: 2rem;
        border-radius: 10px;
    }
    .main-title {
        text-align: center;
        color: #003366;
        font-size: 2.5em;
        font-weight: bold;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = tf.keras.models.load_model("model.h5")

# Load dataset and get feature columns
df = pd.read_csv("Train.csv")
feature_columns = df.drop(columns=['Reached.on.Time_Y.N']).columns

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Load LabelEncoders
le = {}
for col in categorical_cols:
    try:
        le[col] = joblib.load(f"{col}_encoder.pkl")
    except FileNotFoundError:
        st.error(f"Encoder file for '{col}' not found.")
        st.stop()

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Scaler file not found.")
    st.stop()

# Title
st.markdown("<div class='main-title'>🚚 Predict E-Commerce Delivery Timeliness</div>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("📝 Enter Shipment Details")

def user_input_features():
    user_data = {}
    for feature in feature_columns:
        if feature in categorical_cols:
            classes = le[feature].classes_
            if len(classes) == 0:
                st.sidebar.error(f"No available options for {feature}. Please check encoder.")
                st.stop()
            user_data[feature] = st.sidebar.selectbox(f"{feature}", classes)
        else:
            user_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.2f")
    return pd.DataFrame([user_data])

# Get input
data = user_input_features()

# Encode
for col in categorical_cols:
    if data[col][0] in le[col].classes_:
        data[col] = le[col].transform(data[col].astype(str))
    else:
        st.error(f"Invalid value for {col}. Please select from available options.")
        st.stop()

# Scale
data_scaled = scaler.transform(data)

# Predict
if st.sidebar.button("🚀 Predict Delivery Status"):
    prediction = model.predict(data_scaled)
    result = "✅ Delivered on Time" if prediction[0][0] < 0.5 else "⚠️ Delayed"
    st.success(f"### Prediction Result: {result}")
