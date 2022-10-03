from functions.train_pipeline import *
import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
import dill
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.title("Cross-venue convergion")
st.sidebar.title('Selection of tick index and target')


save_path='cache/'
with open(f'{save_path}/df.pkl', 'rb') as out_strm: 
    df = dill.load(out_strm)
with open(f'{save_path}/shap_values.pkl', 'rb') as out_strm: 
    shap_values = dill.load(out_strm)
with open(f'{save_path}/shap_values_.pkl', 'rb') as out_strm: 
    shap_values_ = dill.load(out_strm)
with open(f'{save_path}/explainer.pkl', 'rb') as out_strm: 
    explainer = dill.load(out_strm)
with open(f'{save_path}/train_vars.pkl', 'rb') as out_strm: 
    train_vars = dill.load(out_strm)

# st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)

t_index = st.sidebar.slider('index of lob event time', 
                            min_value=0, 
                            max_value=shap_values_.shape[0], 
                            value = 100, key = 'st_t_index')

    
st.header('Prediction explanation')
st_shap(shap.plots.waterfall(shap_values[t_index]), height=500)
st.header('Prediction explanation aggregated by venue')
st_shap(shap.plots.waterfall(shap_values_[t_index]), height=400)

df_shap = pd.DataFrame(shap_values_.values,columns=shap_values_.feature_names)
df_shap_plot = abs(df_shap.copy())*1000                   

st.header('Force plot venue explanations')
plot = go.Figure(data=[go.Scatter(
    name='venue0_agg',
    x = df.time,
    y = df_shap_plot['venue0_agg'],
    stackgroup='one'),
                        go.Scatter(
    name='venue1_agg',                           
    x = df.time,
    y = df_shap_plot['venue1_agg'],
    stackgroup='two'),
                        go.Scatter(
    name='venue2_agg',                           
    x = df.time,
    y = df_shap_plot['venue2_agg'],
    stackgroup='three'),
                        go.Scatter(
    name='venue3_agg',                           
    x = df.time,
    y = df_shap_plot['venue3_agg'],
    stackgroup='four'),  
                        go.Scatter(
    name='venue4_agg',                           
    x = df.time,
    y = df_shap_plot['venue4_agg'],
    stackgroup='five'),                       
])
st.plotly_chart(plot)