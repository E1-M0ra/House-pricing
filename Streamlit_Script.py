import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_training_data():
    return pd.read_csv("train.csv") 

df = load_training_data()
df.drop(columns=['3SsnPorch', 'MiscVal', 'MoSold', 'Id', 'MSSubClass', 'OverallQual', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtFullBath', 'YrSold', 'GarageArea', 'TotRmsAbvGrd', 'PoolQC', 'RoofMatl', 'Condition2', 'Utilities', 'Street', 'SalePrice'], inplace=True)
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

@st.cache_resource
def load_my_model():
    model = joblib.load("house_price_model.pkl")
    return model

model = load_my_model()

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🔮 Machine Learning Prediction App")
st.write("Enter the required details below to get an instant prediction from the model.")

feature_names = df.columns.tolist()
user_inputs = {}

columns = st.columns(3)

for i, feature in enumerate(feature_names):
    with columns[i % 3]:
        if feature in categorical_features:
            user_inputs[feature] = st.selectbox(f"Select {feature}:", df[feature].unique())
        else:
            min_value = int(df[feature].min())
            max_value = int(df[feature].max())
            user_inputs[feature] = st.slider(f"Select {feature}:", min_value, max_value, (min_value + max_value) // 2)

st.subheader("📋 Input Features")


st.markdown("---")

if st.button("🚀 Run Prediction", type="primary"):
    with st.spinner("Calculating..."):
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[feature_names]
        prediction = model.predict(input_df)
        
        st.success(f"🔮 **Predicted Value:** {prediction[0]:.2f}")
