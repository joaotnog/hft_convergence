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
