from sklearn import datasets
import streamlit as st

@st.cache_data
def load_data():
    data =  datasets.load_breast_cancer(as_frame=True)
    data_frame = data.frame.copy()
    return data_frame

@st.cache_data
def transform_columns(data):
    data_frame = data.copy()
    data_frame.columns = (
        data_frame.columns
            .str.lower()
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.replace(' ', '_')
            .str.replace(r'[^a-z0-9]', '_', regex=True)
    ) 
    return data_frame

@st.cache_data
def main_load():
    df1 = load_data()
    df = transform_columns(df1)
    return df
    