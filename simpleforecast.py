import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost import XGBRegressor
from collections import OrderedDict
import util

def fit(df,auxiliary_variables,forecasting_variable,q,
            steps_ahead=1,models=[RandomForestRegressor],
            variable_selection_type = ['Correlation'],
            manual_list = [],
            variable_lag_selection=False, hyperparameters=[],max_lags=30,data_Itest= '2013-01-01',
            data_Ftest= '2013-12-31',data_Itreino='1993-01-01',data_Ftreino='2011-12-31'):

    """ Recieves multivariate time series data and returns a model

    Args:
        df: Data frame with multivariate time series data
        forecasting_variable: variables to be forecasted
        auxiliary_variables:list of variables to use to aid prediction
        steps_ahead: Forescating horizon
        models: List of candidate models 
        variable_selection_type: Defines the variable selection method. Can be 'Correlation' or 'ExtraTrees'
        hyperparameters: List of dictionaries with hyper parameters for each model. It is used for 
        max_lags: Maximum number of lags in the model

    Return: 
        A fitted model for the time series
    
    """
    # Collects the name of the dataframe columns if the user does not specify
    aux=util.get_column(df,forecasting_variable)
    # Generates a new data frame with max_lags
    dff=util.displace(df,aux["variable_list"],aux["Target"],max_lags)
    # Performs variable selection 
    selected_variables=variable_selection(df,auxiliary_variables,forecasting_variable,q,variable_selection_type,
    data_Itreino,data_Ftreino,max_lags,steps_ahead=1)
    # Performs model selection ()
    selected_model = model_selection(df,forecasting_variable,selected_variables, models=[RandomForestRegressor], manual_list = [],
             hyperparameters=[])

        
       

def variable_selection(df,auxiliary_variables,forecasting_variable,q,steps_ahead=1,
       variable_selection_type = ['Correlation'],max_lags=30,data_Itreino='1993-01-01',
       data_Ftreino='2011-12-31'):
    """ Selects the best lags an the best exogenous variables for the given data
      Args:
             df: Data frame with multivariate time series data
             q: number of ranked attributes.By default that= 10 best ranked attributes
             forecasting_variable: variables to be forecasted
             auxiliary_variables:list of variables to use to aid prediction
             steps_ahead: Forescating horizon
             variable_selection_type: Defines the variable selection method. Can be 'Correlation' or 'feature_importances'
             max_lags: Maximum number of lags in the model
             data_Itreino=Start date for data training
             data_Ftreino=last date for training
        Returns:
               list of lists containing :best lags and best ​​exogenous variables for data provided based on "Correlation" or "feature_importances"
   """
    if auxiliary_variables!=[]:
        # Generates a new data frame with max_lags and indicated variables
       dff=util.displace(df,auxiliary_variables,forecasting_variable,max_lags)
    
    else:
        # Collects the name of the dataframe columns if the user does not specify
        aux=util.get_column(df,forecasting_variable)
        # Generates a new data frame with max_lags and all variables 
        dff=util.displace(df,aux["variable_list"],aux["Target"],max_lags)

    filtered_list=util.column_Filter(dff,steps_ahead,forecasting_variable)
    ranked_list=util.resource_ranking(df,filtered_list,forecasting_variable,data_Itreino,data_Ftreino) 
    standardized_variable_list=util.standardize_variable_list (ranked_list[variable_selection_type],q,forecasting_variable)

    return standardized_variable_list


def model_selection(df,forecasting_variable,selected_variables, models=[RandomForestRegressor],
             manual_list = [],  hyperparameters=[]):
    '''
            Return the results of the model given
    '''
    for i in range (len(models)):
        if models[i] == RandomForestRegressor :
            if hyperparameters[i] !=[]:
                selected_model = util.florestasAleatorias(df,[selected_variables,manual_list],forecasting_variable,hyperparameters[i])
            else:
                selected_model = util.florestasAleatorias(df,[selected_variables,manual_list],forecasting_variable)
        elif models[i] ==DecisionTreeRegressor:
            if hyperparameters[i] !=[]:
                selected_model = util.arvores(df,[selected_variables,manual_list],forecasting_variable,hyperparameters[i])
            else:
                selected_model = util.arvores(df,[selected_variables,manual_list],forecasting_variable)
        elif models[i]==XGBRegressor:
            if hyperparameters[i] !=[]:
                selected_model = util.xgbs(df,[selected_variables,manual_list],forecasting_variable,hyperparameters[i])
            else:
                selected_model = util.xgbs(df,[selected_variables,manual_list],forecasting_variable)

    return selected_model
def finetunnig_models(df,models, params,list,lags,Target,lags_Target,score='neg_mean_squared_error',data_Itreino='1993-01-01',data_Ftreino='2011-12-31'):
    """ Performs a grid search for the input model using the set of hyper parameters 
    
        Args:
            df: Data frame with multivariate time series data
            model: dictionary with the scikit learn model
            hyperparameters: Set of candidate hyperparameters 
            params:Dictionary that contains dictionaries with search spaces for each model
            score:grid search scoring metrics.Por padrao scoring='neg_mean_squared_error'
        Returns:
            best_model:best model of the models informed within the grid search search space
            best_hyperparameters:dictionary with the best hyperparameters found in the search space
            all_results: all model grid search results

    """

    x1_train, x1_test,y1_train, y1_test= util.get_train_test_sets(df, list,lags,Target,lags_Target,data_Itreino,data_Ftreino)
    ### TEST ###

    x=x1_train.values
    y=y1_train.values.ravel()
    print(x.shape,"X",y.shape,"Y")
    # OrderedDict() is recommended here
    # to maintain order between models and params 
    if not bool(models) and not bool(params) :
        models = {
                    'model_xgb': XGBRegressor(objective = "reg:squarederror"),
                    'model_dt': DecisionTreeRegressor(),
                    'model_rf': RandomForestRegressor(),
                    
                    }
    
        params_xgb={
                        'max_depth': [3, 4, 5,6,8],
                        }
        params_dt = {'splitter': ['best', 'random'],
                        'max_depth': [1,3,5]}
        params_rf = {
                        "max_depth" : [1,3,5]}

        # OrderedDict() is recommended here
        # to maintain order between models and params 
        params = OrderedDict()
        params['params_xgb']=params_xgb
        params['params_dt']=params_dt
        params['params_rf']=params_rf
        # print(params["params_xgb"])
        
        best_model, all_results = util.models_gridSearchCV(models, params,score , x, y)
    else:
        best_model, all_results = util.models_gridSearchCV(models, params, score, x, y)

    
    return {
            'best_model':best_model["best_estimator"],
            'best_hyperparameters':best_model['params'],
            "all_results":all_results}

def test(df, model, metrics=['mae'],data_Itest= '2013-01-01',data_Ftest= '2013-12-31'):
 """ Evaluates a model given a test set
    
 """
 erros=[]
 y1_pred = model.predict(x1_test)
 
 std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
 for i in metrics:
    if i == ['mae']:
     mse = mean_squared_error(y1_test, y1_pred)
     std_mse=np.std(np.sqrt((y1_pred - y1_test)**2))
  
    erros.append("std_mse",round(std_mse,2))
    
  
   return {'best_model': best_model,
            'best_hyperparameters': best_hyperparameters}
