# import streamlit as st
# from streamlit_shap import st_shap
# import shap

# from sklearn.model_selection import train_test_split
# import xgboost

# import numpy as np
# import pandas as pd


# @st.experimental_memo
# def load_data():
#     return shap.datasets.adult()

# @st.experimental_memo
# def load_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
#     d_train = xgboost.DMatrix(X_train, label=y_train)
#     d_test = xgboost.DMatrix(X_test, label=y_test)
#     params = {
#         "eta": 0.01,
#         "objective": "binary:logistic",
#         "subsample": 0.5,
#         "base_score": np.mean(y_train),
#         "eval_metric": "logloss",
#         "n_jobs": -1,
#     }
#     model = xgboost.train(params, d_train, 10, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)
#     return model

# st.title("SHAP in Streamlit")
# st.sidebar.title('WalnutTradingDash :chart_with_upwards_trend:')
# st.sidebar.caption('v 1.2.0')

# # train XGBoost model
# X,y = load_data()
# X_display,y_display = shap.datasets.adult(display=True)

# model = load_model(X, y)

# # compute SHAP values
# explainer = shap.Explainer(model, X)
# shap_values = explainer(X)

# st_shap(shap.plots.waterfall(shap_values[0]), height=300)
# st_shap(shap.plots.beeswarm(shap_values), height=300)

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display.iloc[0,:]), height=200, width=1000)
# st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:]), height=400, width=1000)

# exit_data2 = st.sidebar.number_input('Specifyk Input Value', min_value = 0, value = 80, key = 'number2')

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
                   

st.header('Force plot venue explanations')
@st.cache
def force_plot_venues(st, shap_values_, df):
    df_shap = pd.DataFrame(shap_values_.values,columns=shap_values_.feature_names)
    df_shap_plot = abs(df_shap.copy())*1000
      
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
    
force_plot_venues(st, shap_values_, df)
