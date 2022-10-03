import os
import pandas as pd
from functools import reduce
import xgboost
import shap
import numpy as np
import copy
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import dill
from functions.avellaneda_stoikov_model import get_stoikov_prices
from functions.train_pipeline import *
pio.renderers.default='browser'





venues_dict = read_venues()
df = feat_eng(venues_dict)
df = build_target(df, target='midprice',steps_ahead = 5)
train_vars, explainer, model, predictions, shap_values, shap_values_ = train(df, len(venues_dict.keys()))

  
save_path='cache/'
with open(f'{save_path}/df.pkl', 'wb') as out_strm: 
    dill.dump(df, out_strm)
with open(f'{save_path}/shap_values.pkl', 'wb') as out_strm: 
    dill.dump(shap_values, out_strm)
with open(f'{save_path}/shap_values_.pkl', 'wb') as out_strm: 
    dill.dump(shap_values_, out_strm)
with open(f'{save_path}/explainer.pkl', 'wb') as out_strm: 
    dill.dump(explainer, out_strm)
with open(f'{save_path}/train_vars.pkl', 'wb') as out_strm: 
    dill.dump(train_vars, out_strm)
    
df_shap = pd.DataFrame(shap_values_.values, columns=shap_values_.feature_names)


# import plotly.graph_objects as go
# import plotly.express as px
# import numpy
  
  
# df2 = px.data.iris()
# df2 = abs(df_shap.copy())*1000
# # df2 = df_shap.copy()*1000
# # df2 = pd.DataFrame(dict(venue0_agg=[1,5,4],
# #                         venue1_agg=[8,5,3]))
  
# plot = go.Figure(data=[go.Scatter(
#     name='venue0_agg',
#     x = df.time,
#     y = df2['venue0_agg'],
#     stackgroup='one'),
#                        go.Scatter(
#     name='venue1_agg',                           
#     x = df.time,
#     y = df2['venue1_agg'],
#     stackgroup='two'),
#                        go.Scatter(
#     name='venue2_agg',                           
#     x = df.time,
#     y = df2['venue2_agg'],
#     stackgroup='three'),
#                        go.Scatter(
#     name='venue3_agg',                           
#     x = df.time,
#     y = df2['venue3_agg'],
#     stackgroup='four'),  
#                        go.Scatter(
#     name='venue4_agg',                           
#     x = df.time,
#     y = df2['venue4_agg'],
#     stackgroup='five'),                       
# ])
                   
# plot.show()









