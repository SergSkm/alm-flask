#from app import app as appl
# from waitress import serve
from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import requests
app=Flask(__name__)
@app.route('/')
def index():
    return'HI!'




np.random.seed(123)

# @app.route('/get-alm', methods=['POST'])
# def get_ALM_forecasts():
    # post_request_object = request.get_json(force=True)
    # print(post_request_object)

    # return(str(post_request_object))
    
@app.route('/get-alm', methods=['POST'])
def get_ALM_forecasts():

    post_request_object = request.get_json(force=True)
    
    py_response = {}
    
    models_all = ['RidgeCV', 'LassoCV', 'BayesianRidge', 'RandomForestRegressor', 
                  'GradientBoostingRegressor', 'AdaBoostRegressor', 'LightGBM', 'MLPRegressor',
                  'ElasticNetCV', 'CatBoostRegressor', 'KNeighborsRegressor', 'LinearRegression', 'MultiPolynom']
    model1, model2 = "LinearRegression", "LinearRegression"
    
    if post_request_object['model1'] in models_all:
        model1 = post_request_object['model1']
    if post_request_object['model2'] in models_all:
        model2 = post_request_object['model2']
        
    

    rates_delta_forecast, rates_delta_forecast_base, mrktShare_delta_forecast, mrktShare_delta_forecast_base = [], [], [], []

    
#     # Case 1. General forecasts
#     for tenor, rates_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
#                                   np.stack((post_request_object['ftp_delta_shocked'],
#                                             post_request_object['mosprime_delta'],
#                                             post_request_object['limit_delta']), axis=1)):

#         modelFitted1 = pickle.load( open("alm_flask_TEST/models/model1/" + model1 + "/" + \
#                                      post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

#         rates_delta_forecast.append(np.float(modelFitted1.predict(rates_input.reshape(-1,3))))

#     for tenor, mrktShare_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
#                                   np.stack((post_request_object['mosprime_delta'], 
#                                             rates_delta_forecast), axis=1)):

#         modelFitted2 = pickle.load( open("alm_flask_TEST/models/model2/" + model2 + "/" + \
#                                      post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

#         mrktShare_delta_forecast.append(np.float(modelFitted2.predict(mrktShare_input.reshape(-1,2))))
        
#     rates_forecast = np.array(post_request_object['rates_fact']) + np.array(rates_delta_forecast)
#     mrktShare_forecast = np.array(post_request_object['mrktShare_fact']) + np.array(mrktShare_delta_forecast)

#     py_response['rates_delta_forecast'] = list(rates_delta_forecast)
#     py_response['mrktShare_delta_forecast'] = list(mrktShare_delta_forecast)

#     py_response['rates_forecast'] = list(rates_forecast)
#     py_response['mrktShare_forecast'] = adjust_MrktShare_To100(mrktShare_forecast)
    
    
    # Case 2. Difference between shocked case and zer0-shock case is added to the fact!
    for tenor, rates_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['ftp_delta_shocked'],
                                            post_request_object['mosprime_delta'],
                                            post_request_object['limit_delta']), axis=1)):

        modelFitted1 = pickle.load( open("models/model1/" + model1 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        rates_delta_forecast.append(np.float(modelFitted1.predict(rates_input.reshape(-1,3))))

    for tenor, rates_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['ftp_delta'],
                                            post_request_object['mosprime_delta'],
                                            post_request_object['limit_delta']), axis=1)):

        modelFitted1 = pickle.load( open("models/model1/" + model1 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        rates_delta_forecast_base.append(np.float(modelFitted1.predict(rates_input.reshape(-1,3)))) 


    for tenor, mrktShare_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['mosprime_delta'], 
                                            rates_delta_forecast), 
                                           axis=1)):

        base_directory = "models/model2/" + model2 + "/" + post_request_object['sector'] + "_"
        modelFitted2 = pickle.load(open(base_directory + tenor + ".pkl", "rb" )) 

        mrktShare_delta_forecast.append(np.float(modelFitted2.predict(mrktShare_input.reshape(-1,2))))  


    for tenor, mrktShare_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['mosprime_delta'], 
                                            rates_delta_forecast_base), 
                                           axis=1)):

        base_directory = "models/model2/" + model2 + "/" + post_request_object['sector'] + "_"
        modelFitted2 = pickle.load(open(base_directory + tenor + ".pkl", "rb" )) 

        mrktShare_delta_forecast_base.append(np.float(modelFitted2.predict(mrktShare_input.reshape(-1,2))))   

    # adding back differences between shocked case and zero-shock case:
    rates_forecast_difference = np.nan_to_num(np.array(rates_delta_forecast) - np.array(rates_delta_forecast_base))
    #mrktShare_forecast_difference = np.array(mrktShare_delta_forecast) - np.array(mrktShare_delta_forecast_base)
    mrktShare_forecast_difference = np.array(mrktShare_delta_forecast) - np.array(mrktShare_delta_forecast_base)
    #mrktShare_forecast_difference = np.nan_to_num(np.array(adjust_MrktShareDelta_To0(mrktShare_forecast_difference)))
    #________________________________________________#
    # Adjustment function adjust_MrktShare_To100() although monotonous is not smooth enough for the final result!!
    # I should think of improvement!
    #________________________________________________#
    
    
    rates_forecast = np.array(post_request_object['rates_fact']) + rates_forecast_difference
    mrktShare_forecast = np.array(post_request_object['mrktShare_fact']) + mrktShare_forecast_difference


    py_response['rates_delta_forecast'] = list(rates_delta_forecast)
    py_response['mrktShare_delta_forecast'] = list(mrktShare_delta_forecast)

    py_response['rates_forecast'] = list(rates_forecast)
    #py_response['mrktShare_forecast'] = adjust_MrktShare_To100(mrktShare_forecast)
    py_response['mrktShare_forecast'] = list(mrktShare_forecast)
    
    return(str(py_response))
    
def adjust_MrktShare_To100(mrktShare_forecast):
    
    ajdustment_table = []
    
    ajdustment_table = pd.DataFrame(mrktShare_forecast, columns=['value'])
    ajdustment_table['abs'] = ajdustment_table.apply(lambda x: abs(x))

    ajdustment_table.sort_values(by=['abs'], ascending=True, inplace=True)
    ajdustment_table['adjusted'] = ajdustment_table["abs"].cumsum()/sum(ajdustment_table["abs"].cumsum())
    ajdustment_table.sort_index(ascending=True, inplace=True)
    
    return list(ajdustment_table['adjusted'])
    
def adjust_MrktShareDelta_To0(mrktShare_forecast_difference):
    
    weights = []
    weights = adjust_MrktShare_To100(mrktShare_forecast_difference)

    return mrktShare_forecast_difference - np.multiply(weights, mrktShare_forecast_difference)
    
    
if __name__=='__main__':
    app.run(threaded=True, debug=True)  
