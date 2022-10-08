from functions.train_pipeline import *
import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
import dill
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import copy


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

target = st.sidebar.selectbox('Target for aggregated LOB prediction', ['Mid Price','Imbalance Weighted Mid Price','Stoikov Micro Price'])
target_map = {'Mid Price':'midprice',
              'Imbalance Weighted Mid Price':'imbalance_midprice',
              'Stoikov Micro Price':'stoikov_microprice'}

steps_ahead = st.sidebar.slider('Ticks ahead prediction', 
                                min_value=5, 
                                max_value=20, 
                                value = 5, key = 't_ahead')

var1_name = st.sidebar.selectbox('Interaction plot var 1', 
                                 shap_values.feature_names, 
                                 index=shap_values.feature_names.index('venue0_imbalance_lv5'))
var2_name = st.sidebar.selectbox('Interaction plot var 2', 
                                 shap_values.feature_names, 
                                 index=shap_values.feature_names.index('venue0_spread'))




train_bt = st.sidebar.button('Retrain ML', 
                          help='Retrain ML with updated target/time horizon')

st.title(f"ML cross-venue convergion prediction for {target} {steps_ahead} ticks ahead")

if train_bt:
    venues_dict = read_venues()
    df = feat_eng(venues_dict)
    df = build_target(df, target=target_map[target],steps_ahead = steps_ahead)
    train_vars, explainer, model, predictions, shap_values, shap_values_ = train(df, len(venues_dict.keys()))
    
    
st.header(f'Prediction explanation for time index {t_index}')
st.subheader('Explanation of top prediction drivers')
st_shap(shap.plots.waterfall(shap_values[t_index]), height=400)
st.subheader('Explanation of top prediction drivers aggregated by venue')
st_shap(shap.plots.waterfall(shap_values_[t_index]), height=400)

df_shap = pd.DataFrame(shap_values_.values,columns=shap_values_.feature_names)
df_shap_plot = abs(df_shap.copy())*1000                   


st.header('Overall explanation plots')
st.subheader('Venue contribution per time index')

plot = go.Figure(data=[go.Scatter(
    name='venue0_agg',
    x = df.time,
    y = df_shap_plot['venue0_agg'],
    mode='lines',
    line=dict(width=0.5, color='red'),
    stackgroup='one',
    groupnorm='percent'),
                        go.Scatter(
    name='venue1_agg',                           
    x = df.time,
    y = df_shap_plot['venue1_agg'],
    mode='lines',
    line=dict(width=0.5, color='blue'),    
    stackgroup='one'),
                        go.Scatter(
    name='venue2_agg',                           
    x = df.time,
    y = df_shap_plot['venue2_agg'],
    mode='lines',
    line=dict(width=0.5, color='orange'),    
    stackgroup='one'),
                        go.Scatter(
    name='venue3_agg',                           
    x = df.time,
    y = df_shap_plot['venue3_agg'],
    mode='lines',
    line=dict(width=0.5, color='yellow'),
    stackgroup='one'),  
                        go.Scatter(
    name='venue4_agg',                           
    x = df.time,
    y = df_shap_plot['venue4_agg'],
    mode='lines',
    line=dict(width=0.5, color='green'),
    stackgroup='one'),                       
])
plot.update_layout(
    showlegend=True,
    yaxis=dict(
        type='linear',
        range=[1, 100],
        ticksuffix='%'))
st.plotly_chart(plot)


st.subheader('Distribution of top importance drivers')
st_shap(shap.plots.beeswarm(copy.deepcopy(shap_values)), height=300)
st.subheader('Cluster analysis top importance drivers')
np.random.seed(123)
random_index_shap_values = list(np.random.choice(shap_values.shape[0], min(1000, shap_values.shape[1])))
st_shap(shap.plots.heatmap(shap_values[random_index_shap_values], show = False), height=300)
st.subheader('Interaction plot')
# var1_name = 'venue4_imbalance_midprice'
# var2_name = 'venue4_spread'
var1 = shap_values.feature_names.index(var1_name)
var2 = shap_values.feature_names.index(var2_name)
max_outlier_threshold = np.quantile(shap_values[:,var1].data,.99)
min_outlier_threshold = np.quantile(shap_values[:,var1].data,.01)

st_shap(shap.plots.scatter(shap_values[:,var1], 
                   color=shap_values[:,var2], 
                   show = False, 
                   xmax=max_outlier_threshold,
                   xmin=min_outlier_threshold), height=300)
