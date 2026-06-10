import streamlit as st
import os
import pandas as pd
import sys
import plotly.express as px

# Add root directory to python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import config
from src.data.load_data import load_mall_data

st.set_page_config(page_title="Dataset Explorer", layout="wide")

st.title("📊 Mall Customer Dataset Explorer")
st.write("Browse and analyze customer demographics and spending habits before running K-Means.")

if not os.path.exists(config.MALL_DATA_PATH):
    st.warning("Mall dataset not found.")
else:
    df = pd.read_csv(config.MALL_DATA_PATH)
    
    st.subheader("Raw Customer Records Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Attribute Distributions")
        selected_col = st.selectbox("Select Feature to Plot", config.NUMERICAL_COLS)
        fig_hist = px.histogram(df, x=selected_col, nbins=20, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Gender Demographics")
        gender_counts = df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig_pie = px.pie(gender_counts, names="Gender", values="Count", title="Gender Distribution of Mall Shoppers")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.subheader("Income vs Spending Score Scatter Plot")
    fig_scatter = px.scatter(
        df, 
        x="Annual Income (k$)", 
        y="Spending Score (1-100)", 
        color="Gender",
        size="Age",
        title="Annual Income vs Spending Score (Sized by Age)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
