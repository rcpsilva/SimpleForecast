import pandas as pd
from simpleforecast import fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from collections import OrderedDict


path= 'C:/Users/Ray/Documents/GitHub/'
data_Patricia_Eto = pd.read_csv(path+"SimpleForecast/Dados_PP_Eto.csv") 
data_Patricia_Eto = pd.read_csv(path+"SimpleForecast/data_PEto.csv") 
print(data_Patricia_Eto)


# auxiliary_variables=['Tmax', 'Tmin', 'I', 'Tmean', 'UR', 'V', 'J']
# auxiliary_variables=['Tmax', 'Tmin', 'J']
auxiliary_variables=[]
# feature_importances




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

params = OrderedDict()
params['params_xgb']=params_xgb
params['params_dt']=params_dt
params['params_rf']=params_rf
        









resultado = fit(data_Patricia_Eto,"Eto",10,1,'Correlation',30,auxiliary_variables,metrics=['mae','mse'],
data_Itest= '2013-01-01',data_Ftest= '2013-12-31',data_Itreino='1993-01-01',data_Ftreino='2011-12-31'
,models=models,hyperparameters=params
)

print(resultado)


# path= 'C:/Users/Ray/Documents/GitHub/'
# data_INMET_Eto = pd.read_csv(path+"SimpleForecast/data_Ray.csv") 
# data_INMET_Eto=data_INMET_Eto.drop("radiacao",axis=1)
# print(data_INMET_Eto)

# auxiliary_variables=[]

# resultado = fit(data_INMET_Eto,"temp_media",10,1,'Correlation',30,auxiliary_variables,metrics=['mae'],data_Itest= '2019-01-01',
#             data_Ftest= '2019-12-31',data_Itreino='2018-01-01',data_Ftreino='2018-12-31')
#         # fit(df,forecasting_variable,q,steps_ahead=1,
#         #     variable_selection_type = 'Correlation',max_lags=30,auxiliary_variables=False,
#         #     variable_lag_selection=False,data_Itest= '2013-01-01',
#         #     data_Ftest= '2013-12-31',data_Itreino='1993-01-01',data_Ftreino='2011-12-31',score='neg_mean_squared_error', metrics=['mse'],models=[RandomForestRegressor],manual_list = [],hyperparameters=[])

# print(resultado)