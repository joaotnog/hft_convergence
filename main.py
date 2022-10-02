import os
import pandas as pd
from functools import reduce
import xgboost
import shap
import numpy as np
import copy
from functions.avellaneda_stoikov_model import get_stoikov_prices


def get_file_cols():
    orderbook_cols = [['ask_price_'+str(i+1), 'ask_size_'+str(i+1), 'bid_price_'+str(i+1), 'bid_size_'+str(i+1)] 
                      for i in range(5)]
    orderbook_cols = [item for sublist in orderbook_cols for item in sublist]
    message_cols = ['time','type','order_id','size','price','direction']
    return orderbook_cols, message_cols

def read_venues():
    venues_dict = dict()
    venues = os.listdir('data')
    for venue in venues:
        try:
            orderbook_cols, message_cols = get_file_cols()
            orderbook_file = [x for x in os.listdir('data/'+venue) if 'orderbook' in x][0]
            message_file = [x for x in os.listdir('data/'+venue) if 'message' in x][0]
            orderbook = pd.read_csv(f'data/{venue}/{orderbook_file}',header=None)
            orderbook.columns = orderbook_cols
            message = pd.read_csv(f'data/{venue}/{message_file}',header=None)
            message.columns = message_cols  
            lob = pd.concat([orderbook,message],axis=1)
            venues_dict[venue]=lob
        except:
            pass
    return venues_dict

def get_agg_price_x(x, n_venues):
    agg_ask = min(x[[f'venue{i}_ask_price_1' for i in range(n_venues)]])
    agg_bid = max(x[[f'venue{i}_bid_price_1' for i in range(n_venues)]])
    agg_price = (agg_bid+agg_ask)/2
    return agg_price

def feat_eng(venues_dict):
    n_venues = len(venues_dict.keys())
    for i, key in enumerate(venues_dict.keys()):
        print(i,key)
        prices = venues_dict[key].price.tolist()
        time = venues_dict[key].time.tolist()         
        venues_dict[key]['stoikov'] = get_stoikov_prices(prices,time)
        venues_dict[key].columns = [f'venue{i}_{x}' if x!='time' else x  for x in venues_dict[key].columns]
    df = reduce(lambda  left,right: pd.merge_asof(left, 
                                                right, 
                                                on='time', 
                                                direction='nearest'), 
                       list(venues_dict.values()))    
    df['agg_price'] = df.apply(lambda x:get_agg_price_x(x,n_venues), axis=1) 
    df['agg_stoikov'] = get_stoikov_prices(prices = df.agg_price.tolist(),
                                           time = df.time.tolist())
    return df

def build_target(df, target='midprice',steps_ahead = 5):
    if target=='midprice':   
        df['target'] = df.agg_price.shift(steps_ahead)/df.agg_price-1
    if target=='stoikov':   
        df['target'] = df.agg_stoikov.shift(steps_ahead)/df.agg_stoikov-1
    df['target'] = df['target']*1000
    df = df.dropna()
    df = df[df.target!=0]
    return df

def train(df, n_venues):
    train_vars = [x for x in df.columns if x not in ['time','target']]
    X, y = df[train_vars], df['target']
    model = xgboost.XGBRegressor().fit(X, y)
    predictions = model.predict(X)
    # getting overall shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # building venue aggregated shap values
    shap_df = pd.DataFrame(shap_values.values, columns=train_vars)
    for i in range(n_venues):
        venuei_features = [x for x in train_vars if 'venue'+str(i) in x]
        shap_df[f'venue{i}_agg'] = shap_df[venuei_features].sum(axis=1)
    shap_df = shap_df[[f'venue{i}_agg' for i in range(n_venues)]]
    shap_values_ = copy.deepcopy(shap_values)
    shap_values_.feature_names = list(shap_df.columns)
    shap_values_.values = np.array(shap_df)
    return model, predictions, shap_values, shap_values_


venues_dict = read_venues()
df = feat_eng(venues_dict)
df = build_target(df, target='midprice',steps_ahead = 5)
model, predictions, shap_values, shap_values_ = train(df, len(venues_dict.keys()))



shap.plots.waterfall(shap_values_[0])
shap.plots.waterfall(shap_values[0])








